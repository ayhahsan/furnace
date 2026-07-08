// Greedy decoding loop with streaming UTF-8 output.

use crate::model::{KvCache, Model};
use crate::sampler::Sampler;
use crate::tokenizer::Tokenizer;
use anyhow::ensure;
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

/// Sampled decoding with a KV cache: prefill the prompt once, then decode
/// one token per step, choosing each via the sampler (which degenerates to
/// exact greedy at temperature 0). Streams text to stdout, reports ids and
/// timing on stderr, and returns the generated ids.
pub fn generate(
    model: &Model,
    tokenizer: &Tokenizer,
    prompt_ids: Vec<u32>,
    max_new: usize,
    stop_ids: &[u32],
    sampler: &mut Sampler,
) -> Result<Vec<u32>> {
    let max_ctx = model.config.context_length.min(4096);
    ensure!(
        prompt_ids.len() + max_new <= max_ctx,
        "prompt ({}) + max tokens ({}) exceeds context cap {}",
        prompt_ids.len(), max_new, max_ctx
    );
    let mut cache = KvCache::new(&model.config, max_ctx);
    eprintln!(
        "kv cache: {:.0} MB for {} positions",
        cache.bytes() as f64 / (1024.0 * 1024.0),
        max_ctx
    );

    let mut generated = Vec::new();
    let mut context = prompt_ids.clone();
    let mut stream = Utf8Stream::new();
    let mut stdout = std::io::stdout();

    let t0 = Instant::now();
    let mut logits = model.forward_cached(&prompt_ids, &mut cache);
    let prefill = t0.elapsed().as_secs_f64();
    eprintln!(
        "prefill: {} tokens in {:.2} s ({:.0} ms/token)",
        prompt_ids.len(),
        prefill,
        prefill * 1000.0 / prompt_ids.len() as f64
    );

    crate::perf::reset(); // breakdown covers the decode phase only
    let mut decode_seconds = 0.0;
    let mut decode_steps = 0usize;
    for step in 0..max_new {
        let window = context.len().saturating_sub(sampler.repeat_window);
        let next = sampler.sample(&logits.data, &context[window..]);
        if stop_ids.contains(&next) {
            break;
        }
        generated.push(next);
        context.push(next);
        print!("{}", stream.push(&tokenizer.token_bytes(next)?));
        stdout.flush()?;
        if step + 1 == max_new {
            break;
        }
        let t0 = Instant::now();
        logits = model.forward_cached(&[next], &mut cache);
        let dt = t0.elapsed().as_secs_f64();
        decode_seconds += dt;
        decode_steps += 1;
        eprintln!("[decode {:3}: seq {:4}, {:6.0} ms/token]", step, cache.len, dt * 1000.0);
    }
    print!("{}", stream.flush());
    println!();
    stdout.flush()?;

    if decode_steps > 0 {
        eprintln!(
            "decode: {} steps, {:.0} ms/token average (M6 baseline: 27000-35000 ms/token)",
            decode_steps,
            decode_seconds * 1000.0 / decode_steps as f64
        );
    }
    crate::perf::report("decode", decode_seconds * 1000.0);
    eprintln!("sampler: {} RNG draws", sampler.draws());
    eprintln!("generated ids: {:?}", generated);
    Ok(generated)
}

/// M6 reference path: full recompute every step, no cache, pure argmax.
/// Kept as the ground truth the cached path must match byte for byte.
pub fn generate_uncached(
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
