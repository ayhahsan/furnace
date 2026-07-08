mod generate;
mod gguf;
mod model;
mod perf;
mod quant;
mod sampler;
mod selftest;
mod tensor;
mod tokenizer;

use anyhow::{ensure, Context, Result};
use std::io::{Read, Write};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "furnace", about = "A minimal CPU LLM inference engine")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate text (greedy at --temp 0, sampled otherwise)
    Run {
        /// Path to the .gguf model file
        model: PathBuf,
        /// User prompt
        #[arg(short, long)]
        prompt: String,
        /// Maximum new tokens to generate
        #[arg(short = 'n', long, default_value_t = 50)]
        max_tokens: usize,
        /// Feed the prompt as-is instead of wrapping it in the chat template
        #[arg(long)]
        raw: bool,
        /// Use the M6 full-recompute path instead of the KV cache
        /// (reference implementation; slow, always pure greedy)
        #[arg(long)]
        no_cache: bool,
        /// Sampling temperature; 0 = greedy (deterministic argmax)
        #[arg(long, default_value_t = 0.7)]
        temp: f32,
        /// Keep only the k most likely tokens; 0 disables
        #[arg(long, default_value_t = 40)]
        top_k: usize,
        /// Nucleus sampling: keep the smallest set with cumulative
        /// probability >= p; 1.0 disables
        #[arg(long, default_value_t = 0.9)]
        top_p: f32,
        /// Penalty for tokens seen in the last 64 positions; 1.0 disables
        /// (default off so temp 0 exactly matches greedy)
        #[arg(long, default_value_t = 1.0)]
        repeat_penalty: f32,
        /// RNG seed; omit for one drawn from the clock
        #[arg(long)]
        seed: Option<u64>,
        /// Print a per-op time breakdown of the decode phase
        #[arg(long)]
        timings: bool,
    },
    /// Parse a GGUF file and dump its metadata and tensor table
    Inspect {
        /// Path to the .gguf model file
        model: PathBuf,
        /// Also dump all metadata key-value pairs
        #[arg(long)]
        metadata: bool,
    },
    /// Dequantize a tensor and print values as f32, one per line
    DumpTensor {
        /// Path to the .gguf model file
        model: PathBuf,
        /// Tensor name, e.g. blk.0.attn_q.weight
        tensor: String,
        /// Number of values to print
        #[arg(long, default_value_t = 10)]
        count: usize,
        /// Flat element index to start from (ggml order, dims[0] fastest)
        #[arg(long, default_value_t = 0)]
        offset: usize,
    },
    /// Encode text to token ids (space-separated on one line)
    Tokenize {
        /// Path to the .gguf model file
        model: PathBuf,
        /// Text to encode; omit to read exact bytes from stdin
        #[arg(short, long)]
        prompt: Option<String>,
    },
    /// Decode comma-separated token ids back to text
    Detokenize {
        /// Path to the .gguf model file
        model: PathBuf,
        /// Token ids, e.g. --ids 9707,11,1879 (empty string decodes nothing)
        #[arg(long)]
        ids: String,
    },
    /// Run tensor ops against a reference file from scripts/check_m4.py
    #[command(hide = true)]
    SelftestM4 {
        /// Path to the FTEN reference tensor file
        file: PathBuf,
    },
    /// Staged block-0 comparison against a reference file from check_m5.py
    #[command(hide = true)]
    SelftestM5 {
        /// Path to the .gguf model file
        model: PathBuf,
        /// Path to the FTEN reference tensor file
        file: PathBuf,
    },
    /// Full-forward logits comparison against a reference from check_m6.py
    #[command(hide = true)]
    SelftestM6 {
        /// Path to the .gguf model file
        model: PathBuf,
        /// Path to the FTEN reference tensor file
        file: PathBuf,
    },
}

fn dump_tensor(model: &PathBuf, name: &str, offset: usize, count: usize) -> Result<()> {
    let file = gguf::GgufFile::open(model)?;
    let (info, raw) = file.tensor(name)?;
    let n = info.n_elements();
    let end = offset
        .checked_add(count)
        .filter(|&end| end <= n)
        .with_context(|| {
            format!("range [{}, {}+{}) out of bounds, tensor has {} elements", offset, offset, count, n)
        })?;

    // Dequantize only the blocks covering [offset, end), not the whole tensor.
    let (block_elems, block_bytes) = info
        .dtype
        .block_layout()
        .with_context(|| format!("dtype {:?} not supported yet", info.dtype))?;
    let first_block = offset / block_elems;
    let last_block = end.div_ceil(block_elems);
    ensure!(count > 0, "count must be at least 1");

    let raw_slice = &raw[first_block * block_bytes..last_block * block_bytes];
    let values = quant::dequantize(
        info.dtype,
        raw_slice,
        (last_block - first_block) * block_elems,
    )?;

    let local = offset - first_block * block_elems;
    for v in &values[local..local + count] {
        println!("{}", v);
    }
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Inspect { model, metadata } => {
            let file = gguf::GgufFile::open(&model)?;
            file.dump(metadata);
        }
        Command::DumpTensor { model, tensor, count, offset } => {
            dump_tensor(&model, &tensor, offset, count)?;
        }
        Command::Tokenize { model, prompt } => {
            let text = match prompt {
                Some(p) => p,
                None => {
                    let mut buf = Vec::new();
                    std::io::stdin().read_to_end(&mut buf)?;
                    String::from_utf8(buf).context("stdin is not valid UTF-8")?
                }
            };
            let file = gguf::GgufFile::open(&model)?;
            let tok = tokenizer::Tokenizer::from_gguf(&file)?;
            let ids: Vec<String> = tok.encode(&text)?.iter().map(|id| id.to_string()).collect();
            println!("{}", ids.join(" "));
        }
        Command::Detokenize { model, ids } => {
            let ids = ids
                .split(',')
                .filter(|s| !s.trim().is_empty())
                .map(|s| s.trim().parse::<u32>().with_context(|| format!("bad token id '{}'", s)))
                .collect::<Result<Vec<_>>>()?;
            let file = gguf::GgufFile::open(&model)?;
            let tok = tokenizer::Tokenizer::from_gguf(&file)?;
            let text = tok.decode(&ids)?;
            // write raw UTF-8 bytes plus exactly one trailing newline, so
            // callers can recover the exact decoded string
            let mut out = std::io::stdout();
            out.write_all(text.as_bytes())?;
            out.write_all(b"\n")?;
        }
        Command::SelftestM4 { file } => {
            selftest::run_m4(&file)?;
        }
        Command::SelftestM5 { model, file } => {
            selftest::run_m5(&model, &file)?;
        }
        Command::SelftestM6 { model, file } => {
            selftest::run_m6(&model, &file)?;
        }
        Command::Run { model, prompt, max_tokens, raw, no_cache, temp, top_k, top_p, repeat_penalty, seed, timings } => {
            if timings {
                perf::enable();
            }
            let t0 = std::time::Instant::now();
            let file = gguf::GgufFile::open(&model)?;
            let tok = tokenizer::Tokenizer::from_gguf(&file)?;
            let m = model::Model::load(&file)?;
            eprintln!(
                "loaded {:.0} MB of f32 weights in {:.1} s (dequantized up front; M10 moves this on the fly)",
                m.param_bytes() as f64 / (1024.0 * 1024.0),
                t0.elapsed().as_secs_f64()
            );

            let text = if raw { prompt } else { generate::chat_template(&prompt) };
            let ids = tok.encode(&text)?;
            eprintln!("prompt: {} tokens", ids.len());

            // stop on <|im_end|> (eos) and <|endoftext|> (bos doubles as the
            // pretraining document separator for qwen2)
            let stop_ids = [
                file.meta_u32("tokenizer.ggml.eos_token_id")?,
                file.meta_u32("tokenizer.ggml.bos_token_id")?,
            ];
            if no_cache {
                generate::generate_uncached(&m, &tok, ids, max_tokens, &stop_ids)?;
            } else {
                let seed = seed.unwrap_or_else(|| {
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_nanos() as u64)
                        .unwrap_or(0)
                        ^ std::process::id() as u64
                });
                eprintln!(
                    "sampler: temp {} top-k {} top-p {} repeat-penalty {} seed {}",
                    temp, top_k, top_p, repeat_penalty, seed
                );
                let mut sampler = sampler::Sampler::new(temp, top_k, top_p, repeat_penalty, seed);
                generate::generate(&m, &tok, ids, max_tokens, &stop_ids, &mut sampler)?;
            }
        }
    }
    Ok(())
}
