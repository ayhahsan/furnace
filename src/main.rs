mod gguf;
mod quant;

use anyhow::{ensure, Context, Result};
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
    }
    Ok(())
}
