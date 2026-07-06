// Byte-level BPE tokenizer, loaded entirely from GGUF metadata.
//
// Encode pipeline:
//   1. split out special tokens (matched literally, never touched by BPE)
//   2. pre-tokenize each remaining span with the qwen2 regex
//   3. per chunk: bytes -> GPT-2 unicode alphabet -> merge by rank -> ids
// Decode reverses: ids -> vocab strings -> reverse byte map -> UTF-8 (lossy).

use crate::gguf::GgufFile;
use anyhow::{ensure, Context, Result};
use std::collections::HashMap;

/// Pre-tokenization pattern for tokenizer.ggml.pre == "qwen2", identical to
/// the one in Qwen's tokenizer.json and llama.cpp. The (?!\S) lookahead is
/// why we need fancy-regex.
const QWEN2_SPLIT_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

// tokenizer.ggml.token_type values (gguf spec)
const TOKEN_TYPE_CONTROL: i64 = 3;
const TOKEN_TYPE_USER_DEFINED: i64 = 4;

pub struct Tokenizer {
    /// id -> token string, in the mapped unicode alphabet.
    vocab: Vec<String>,
    token_to_id: HashMap<String, u32>,
    /// "left right" -> rank; lower rank merges first.
    merge_ranks: HashMap<String, u32>,
    /// Special (control / user-defined) tokens, longest literal first.
    specials: Vec<(String, u32)>,
    is_special: Vec<bool>,
    splitter: fancy_regex::Regex,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
}

/// The GPT-2 byte <-> unicode bijection. Printable non-space chars map to
/// themselves; the remaining 68 bytes are assigned codepoints 256, 257, ...
/// in byte order (so space 0x20 -> U+0120 'G-dot', newline 0x0A -> U+010A).
fn byte_unicode_map() -> ([char; 256], HashMap<char, u8>) {
    let mut byte_to_char = ['\0'; 256];
    let mut assigned = [false; 256];
    for b in (0x21..=0x7Eusize).chain(0xA1..=0xACusize).chain(0xAE..=0xFFusize) {
        byte_to_char[b] = char::from_u32(b as u32).unwrap();
        assigned[b] = true;
    }
    let mut next = 256u32;
    for b in 0..256 {
        if !assigned[b] {
            byte_to_char[b] = char::from_u32(next).unwrap();
            next += 1;
        }
    }
    let char_to_byte = byte_to_char
        .iter()
        .enumerate()
        .map(|(b, &c)| (c, b as u8))
        .collect();
    (byte_to_char, char_to_byte)
}

enum Segment<'a> {
    Text(&'a str),
    Special(u32),
}

impl Tokenizer {
    pub fn from_gguf(file: &GgufFile) -> Result<Tokenizer> {
        let model = file.meta_str("tokenizer.ggml.model")?;
        ensure!(model == "gpt2", "tokenizer model '{}' not supported (only gpt2 byte-level BPE)", model);
        let pre = file.meta_str("tokenizer.ggml.pre")?;
        ensure!(pre == "qwen2", "pre-tokenizer '{}' not supported (only qwen2)", pre);

        let tokens = file
            .meta_array("tokenizer.ggml.tokens")?
            .iter()
            .map(|v| v.as_str().map(str::to_owned).context("token is not a string"))
            .collect::<Result<Vec<_>>>()?;
        let token_types = file
            .meta_array("tokenizer.ggml.token_type")?
            .iter()
            .map(|v| v.as_i64().context("token_type is not an integer"))
            .collect::<Result<Vec<_>>>()?;
        let merges = file
            .meta_array("tokenizer.ggml.merges")?
            .iter()
            .map(|v| v.as_str().map(str::to_owned).context("merge is not a string"))
            .collect::<Result<Vec<_>>>()?;

        Tokenizer::new(tokens, &token_types, &merges)
    }

    fn new(tokens: Vec<String>, token_types: &[i64], merges: &[String]) -> Result<Tokenizer> {
        ensure!(
            token_types.len() == tokens.len(),
            "{} tokens but {} token types",
            tokens.len(),
            token_types.len()
        );

        let token_to_id: HashMap<String, u32> = tokens
            .iter()
            .enumerate()
            .map(|(id, t)| (t.clone(), id as u32))
            .collect();
        let merge_ranks: HashMap<String, u32> = merges
            .iter()
            .enumerate()
            .map(|(rank, m)| (m.clone(), rank as u32))
            .collect();

        let is_special: Vec<bool> = token_types
            .iter()
            .map(|&t| t == TOKEN_TYPE_CONTROL || t == TOKEN_TYPE_USER_DEFINED)
            .collect();
        let mut specials: Vec<(String, u32)> = tokens
            .iter()
            .enumerate()
            .filter(|&(id, _)| is_special[id])
            .map(|(id, t)| (t.clone(), id as u32))
            .collect();
        // longest first so that at equal start positions the longest literal wins
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        let splitter = fancy_regex::Regex::new(QWEN2_SPLIT_PATTERN)
            .context("pre-tokenizer pattern failed to compile")?;
        let (byte_to_char, char_to_byte) = byte_unicode_map();

        Ok(Tokenizer {
            vocab: tokens,
            token_to_id,
            merge_ranks,
            specials,
            is_special,
            splitter,
            byte_to_char,
            char_to_byte,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        for segment in self.split_specials(text) {
            match segment {
                Segment::Special(id) => ids.push(id),
                Segment::Text(span) => self.encode_ordinary(span, &mut ids)?,
            }
        }
        Ok(ids)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut bytes = Vec::new();
        for &id in ids {
            bytes.extend_from_slice(&self.token_bytes(id)?);
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Raw bytes of one token. May be a partial UTF-8 sequence; callers that
    /// stream must buffer until a complete character is available.
    pub fn token_bytes(&self, id: u32) -> Result<Vec<u8>> {
        let token = self
            .vocab
            .get(id as usize)
            .with_context(|| format!("token id {} out of range", id))?;
        let mut bytes = Vec::with_capacity(token.len());
        if self.is_special[id as usize] {
            // special tokens are stored as literal text, not byte-mapped
            bytes.extend_from_slice(token.as_bytes());
        } else {
            for c in token.chars() {
                match self.char_to_byte.get(&c) {
                    Some(&b) => bytes.push(b),
                    None => bytes.extend_from_slice(c.to_string().as_bytes()),
                }
            }
        }
        Ok(bytes)
    }

    /// Split text into literal special-token matches (leftmost, longest) and
    /// the plain spans between them.
    fn split_specials<'a>(&self, text: &'a str) -> Vec<Segment<'a>> {
        let mut segments = Vec::new();
        let mut pos = 0;
        while pos < text.len() {
            let mut earliest: Option<(usize, usize, u32)> = None;
            for (literal, id) in &self.specials {
                if let Some(found) = text[pos..].find(literal.as_str()) {
                    let start = pos + found;
                    // strict < keeps the longest literal at equal positions,
                    // because specials are sorted longest first
                    if earliest.map_or(true, |(s, _, _)| start < s) {
                        earliest = Some((start, literal.len(), *id));
                    }
                }
            }
            match earliest {
                Some((start, len, id)) => {
                    if start > pos {
                        segments.push(Segment::Text(&text[pos..start]));
                    }
                    segments.push(Segment::Special(id));
                    pos = start + len;
                }
                None => {
                    segments.push(Segment::Text(&text[pos..]));
                    break;
                }
            }
        }
        segments
    }

    fn encode_ordinary(&self, text: &str, ids: &mut Vec<u32>) -> Result<()> {
        let mut last = 0;
        for m in self.splitter.find_iter(text) {
            let m = m.context("pre-tokenizer regex failed")?;
            // the pattern covers all text, but treat a gap as its own chunk
            // rather than silently dropping bytes
            if m.start() > last {
                self.bpe_chunk(&text[last..m.start()], ids)?;
            }
            self.bpe_chunk(m.as_str(), ids)?;
            last = m.end();
        }
        if last < text.len() {
            self.bpe_chunk(&text[last..], ids)?;
        }
        Ok(())
    }

    /// Run the merge loop on one pre-tokenized chunk.
    fn bpe_chunk(&self, chunk: &str, ids: &mut Vec<u32>) -> Result<()> {
        let mut word: Vec<String> = chunk
            .bytes()
            .map(|b| self.byte_to_char[b as usize].to_string())
            .collect();

        while word.len() > 1 {
            // find the adjacent pair with the lowest merge rank
            let mut best: Option<(u32, usize)> = None;
            for i in 0..word.len() - 1 {
                let key = format!("{} {}", word[i], word[i + 1]);
                if let Some(&rank) = self.merge_ranks.get(&key) {
                    if best.map_or(true, |(r, _)| rank < r) {
                        best = Some((rank, i));
                    }
                }
            }
            let Some((_, idx)) = best else { break };
            let (a, b) = (word[idx].clone(), word[idx + 1].clone());
            let merged = format!("{}{}", a, b);

            // merge every occurrence of this pair, left to right
            let mut next = Vec::with_capacity(word.len());
            let mut i = 0;
            while i < word.len() {
                if i + 1 < word.len() && word[i] == a && word[i + 1] == b {
                    next.push(merged.clone());
                    i += 2;
                } else {
                    next.push(word[i].clone());
                    i += 1;
                }
            }
            word = next;
        }

        for symbol in &word {
            let id = self
                .token_to_id
                .get(symbol)
                .with_context(|| format!("symbol {:?} not in vocab", symbol))?;
            ids.push(*id);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic tokenizer: all 256 byte tokens (id == byte value), three
    /// merges building "hell", and one special token. Round-tripping is a
    /// structural property of byte-level BPE, so it holds even for this toy
    /// vocab; matching real Qwen ids is the checkpoint's job.
    fn tiny() -> Tokenizer {
        let (byte_to_char, _) = byte_unicode_map();
        let mut tokens: Vec<String> = (0..256).map(|b| byte_to_char[b].to_string()).collect();
        let mut types = vec![1i64; 256];
        for t in ["he", "ll", "hell"] {
            tokens.push(t.to_string());
            types.push(1);
        }
        tokens.push("<|test|>".to_string());
        types.push(3);
        let merges: Vec<String> =
            ["h e", "l l", "he ll"].iter().map(|s| s.to_string()).collect();
        Tokenizer::new(tokens, &types, &merges).unwrap()
    }

    fn round_trip(t: &Tokenizer, s: &str) {
        let ids = t.encode(s).unwrap();
        assert_eq!(t.decode(&ids).unwrap(), s, "round trip failed for {:?}", s);
    }

    #[test]
    fn merges_apply_by_rank() {
        let t = tiny();
        // h+e -> he (rank 0), l+l -> ll (rank 1), he+ll -> hell (rank 2),
        // 'o' stays a single byte token (id = byte value 111)
        assert_eq!(t.encode("hello").unwrap(), vec![258, 111]);
    }

    #[test]
    fn ascii_round_trip() {
        round_trip(&tiny(), "Hello, world! 123");
    }

    #[test]
    fn hindi_round_trip() {
        round_trip(&tiny(), "\u{928}\u{92e}\u{938}\u{94d}\u{924}\u{947} \u{926}\u{941}\u{928}\u{93f}\u{92f}\u{93e}");
    }

    #[test]
    fn emoji_round_trip() {
        round_trip(&tiny(), "pizza \u{1F355} and \u{1F363}");
    }

    #[test]
    fn whitespace_round_trip() {
        round_trip(&tiny(), "  leading, trailing  ");
        round_trip(&tiny(), "a   b\t\tc\n\nd");
    }

    #[test]
    fn special_token_is_not_split() {
        let t = tiny();
        let ids = t.encode("a<|test|>b").unwrap();
        assert_eq!(ids, vec![b'a' as u32, 259, b'b' as u32]);
        assert_eq!(t.decode(&ids).unwrap(), "a<|test|>b");
    }

    #[test]
    fn empty_string() {
        let t = tiny();
        assert_eq!(t.encode("").unwrap(), Vec::<u32>::new());
        assert_eq!(t.decode(&[]).unwrap(), "");
    }

    #[test]
    fn chat_template_matches_hf_apply_chat_template() {
        // expected ids come from HF AutoTokenizer.apply_chat_template for
        // [{role: user, content: "Hi"}] with add_generation_prompt=True,
        // captured once (scripts/check_m6.py re-verifies the template string)
        let path = std::path::Path::new("../models/qwen2.5-0.5b-instruct-q8_0.gguf");
        if !path.exists() {
            eprintln!("skipped: model file not present");
            return;
        }
        let file = crate::gguf::GgufFile::open(path).unwrap();
        let tok = Tokenizer::from_gguf(&file).unwrap();
        let ids = tok.encode(&crate::generate::chat_template("Hi")).unwrap();
        assert_eq!(
            ids,
            vec![
                151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553,
                54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645,
                198, 151644, 872, 198, 13048, 151645, 198, 151644, 77091, 198
            ]
        );
    }

    #[test]
    fn byte_map_is_a_bijection() {
        let (byte_to_char, char_to_byte) = byte_unicode_map();
        assert_eq!(char_to_byte.len(), 256);
        for b in 0..=255u8 {
            assert_eq!(char_to_byte[&byte_to_char[b as usize]], b);
        }
        // the two famous ones
        assert_eq!(byte_to_char[0x20], '\u{120}'); // space -> G-dot
        assert_eq!(byte_to_char[0x0A], '\u{10A}'); // newline -> C-dot
    }
}
