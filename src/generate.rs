// Greedy decoding loop with streaming UTF-8 output.

use crate::model::Model;
use crate::tokenizer::Tokenizer;
use anyhow::Result;
use std::io::Write;
use std::time::Instant;

/// Qwen2.5 ChatML template with the model's default system message.
/// Verified to tokenize identically to HF apply_chat_template.
pub fn chat_template(prompt: &str) -> String {
    format!(
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. \
         You are a helpful assistant.<|im_end|>\n\
         <|im_start|>user\n{}<|im_end|>\n\
         <|im_start|>assistant\n",
        prompt
    )
}

/// Buffers token bytes and yields only complete UTF-8 characters, since one
/// token can end mid-character (multibyte scripts hit this constantly).
pub struct Utf8Stream {
    buf: Vec<u8>,
}

impl Utf8Stream {
    pub fn new() -> Utf8Stream {
        Utf8Stream { buf: Vec::new() }
    }

    /// Append bytes; return whatever prefix is now printable.
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.buf.extend_from_slice(bytes);
        match std::str::from_utf8(&self.buf) {
            Ok(s) => {
                let out = s.to_string();
                self.buf.clear();
                out
            }
            Err(e) if e.error_len().is_none() => {
                // incomplete trailing character: emit the valid prefix,
                // keep the partial bytes for the next push
                let valid = e.valid_up_to();
                let out = String::from_utf8(self.buf[..valid].to_vec()).unwrap();
                self.buf.drain(..valid);
                out
            }
            Err(_) => {
                // genuinely invalid sequence: emit lossily and reset
                let out = String::from_utf8_lossy(&self.buf).into_owned();
                self.buf.clear();
                out
            }
        }
    }

    /// Emit any remaining partial bytes (lossily) at end of generation.
    pub fn flush(&mut self) -> String {
        let out = String::from_utf8_lossy(&self.buf).into_owned();
        self.buf.clear();
        out
    }
}

/// Index of the largest value, first occurrence on ties (matches torch).
pub fn argmax(data: &[f32]) -> u32 {
    let mut best = 0;
    for i in 1..data.len() {
        if data[i] > data[best] {
            best = i;
        }
    }
    best as u32
}

/// Greedy decoding: forward -> argmax -> append -> repeat, full recompute
/// every step (the KV cache is M7; per-token timing here is its baseline).
/// Streams text to stdout, reports ids and timing on stderr, and returns
/// the generated ids.
pub fn generate(
    model: &Model,
    tokenizer: &Tokenizer,
    prompt_ids: Vec<u32>,
    max_new: usize,
    stop_ids: &[u32],
) -> Result<Vec<u32>> {
    let mut ids = prompt_ids;
    let mut generated = Vec::new();
    let mut stream = Utf8Stream::new();
    let mut stdout = std::io::stdout();
    let started = Instant::now();

    for step in 0..max_new {
        let t0 = Instant::now();
        let logits = model.forward_last(&ids);
        let next = argmax(&logits.data);
        let dt = t0.elapsed().as_secs_f64();
        eprintln!("[step {:3}: seq {:3}, {:6.2} s/token]", step, ids.len(), dt);
        if stop_ids.contains(&next) {
            break;
        }
        generated.push(next);
        ids.push(next);
        print!("{}", stream.push(&tokenizer.token_bytes(next)?));
        stdout.flush()?;
    }
    print!("{}", stream.flush());
    println!();
    stdout.flush()?;

    let total = started.elapsed().as_secs_f64();
    eprintln!(
        "generated {} tokens in {:.1} s ({:.2} s/token, full recompute -- M7 baseline)",
        generated.len(),
        total,
        total / generated.len().max(1) as f64
    );
    eprintln!("generated ids: {:?}", generated);
    Ok(generated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf8_stream_holds_partial_characters() {
        // Devanagari NA is 3 bytes: E0 A4 A8. Split across pushes, nothing
        // prints until the final byte arrives.
        let mut s = Utf8Stream::new();
        assert_eq!(s.push(&[0xE0, 0xA4]), "");
        assert_eq!(s.push(&[0xA8]), "\u{928}");
        assert_eq!(s.flush(), "");
    }

    #[test]
    fn utf8_stream_mixed_ascii_and_partial() {
        let mut s = Utf8Stream::new();
        // "ab" + first byte of a 2-byte char: ascii prints, partial waits
        assert_eq!(s.push(&[b'a', b'b', 0xC3]), "ab");
        assert_eq!(s.push(&[0xA9, b'!']), "\u{e9}!"); // e-acute
    }

    #[test]
    fn argmax_first_max_wins_ties() {
        assert_eq!(argmax(&[1.0, 5.0, 5.0, 2.0]), 1);
        assert_eq!(argmax(&[-3.0, -1.0, -2.0]), 1);
    }
}
