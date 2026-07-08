# furnace

A minimal LLM inference engine in Rust, written from scratch as a learning
project. Loads a GGUF model file and generates text on CPU. No ML framework:
the GGUF parser, dequantization, BPE tokenizer, tensor ops, transformer,
KV cache, sampler, and fused quantized matmul kernels are all hand-written
and each was verified against a reference implementation before the next
layer was built on top of it.

Dependencies: memmap2, anyhow, clap, half, rayon, fancy-regex (tokenizer
pre-splitting only; the pattern needs lookahead). Nothing else.

```
furnace run model.gguf -p "What is the capital of France?" -n 50
The capital of France is Paris.
```

## What it supports

- GGUF v2/v3, mmap-based, zero-copy: quantized weights are used directly
  from the mapped file, never materialized as f32 (resident memory = file
  size; a 1.5B model runs in ~1 GB of RAM)
- Quantization formats: Q8_0, Q5_0, Q4_K, Q6_K, F32 (fused
  dequantize-inside-the-matmul kernels for all quantized formats)
- Qwen2-architecture models (tested: Qwen2.5-0.5B and 1.5B Instruct);
  config, shapes, and special tokens are read from GGUF metadata
- Byte-level BPE tokenizer loaded from the GGUF (no tokenizer files)
- KV-cached decoding with prefill/decode phases, chat template, streaming
  UTF-8 output (multibyte-safe), temperature / top-k / top-p / repeat
  penalty sampling with a seeded deterministic PRNG

## CLI

```
furnace run <model.gguf> -p <prompt> [-n max] [--temp T] [--top-k K]
            [--top-p P] [--repeat-penalty R] [--seed S] [--raw]
            [--timings] [--f32] [--no-cache]
furnace inspect <model.gguf> [--metadata]
furnace dump-tensor <model.gguf> <tensor> [--count N] [--offset K]
furnace tokenize <model.gguf> [-p text]      (stdin if -p omitted)
furnace detokenize <model.gguf> --ids 1,2,3
```

`--temp 0` is exact greedy decoding (no RNG state is touched). `--f32` and
`--no-cache` select the slower reference implementations that the fast
paths are verified against.

## Numbers

Measured on a Ryzen 7 5700U laptop (8c/16t) with 16 GB DDR4-3200 in a
single module (single-channel, ~21 GB/s practical). CPU was pinned at its
1.8 GHz base clock during the final session; ratios matter more than
absolutes. Decode = ms per generated token, 39-step greedy run.

| model | weights resident | load | decode |
|---|---|---|---|
| 0.5B Q8_0, fused kernels | 639 MB | 0.5 s | 98-106 ms/token |
| 0.5B Q8_0, `--f32` reference | 2404 MB | 3.0 s | 169 ms/token |
| 0.5B Q4_K_M | 463 MB | 0.4 s | 167 ms/token |
| 1.5B Q4_K_M | 1060 MB | 0.4 s | 354 ms/token |

The journey for the 0.5B model: 27,000-35,000 ms/token (first working
forward pass, full recompute) to ~100 ms/token (KV cache + accumulator-lane
dot product + rayon + fused quantized kernels), roughly 300x.

## Milestones

Each milestone ended with a checkpoint against an external reference that
had to pass before the next began.

| milestone | checkpoint |
|---|---|
| M1 GGUF parser | all 291 tensors match the gguf pip package (name/dims/dtype/offset) |
| M2 Q8_0 dequant | bit-exact vs gguf.quants.dequantize, start and mid-tensor |
| M3 BPE tokenizer | 18/18 diverse strings match HF AutoTokenizer; all round-trip |
| M4 tensor ops | matmul/rmsnorm/softmax/swiglu/add/mul vs PyTorch, worst 4.8e-7 |
| M5 one transformer block | six staged intermediates vs hooked HF forward, block out 2.8e-5 |
| M6 full forward + greedy | logits 7e-5, argmax+top-5 exact; greedy matches HF 7/7 tokens |
| M7 KV cache | byte-identical output to the uncached path; decode flat in context length |
| M8 sampling | temp 0 byte-identical to greedy with 0 RNG draws; seed-deterministic |
| M9 performance | 365 -> 117 ms/token, measured AT the memory-bandwidth floor; SIMD skipped with evidence |
| M10 quantized compute | Q4_K/Q6_K/Q5_0 bit-exact; 1.5B logits vs HF 3.3e-5, argmax+top-5 exact |

Verification scripts live in `scripts/` (Python; need `gguf`, `torch`,
`transformers`). The `--timings` flag prints a per-op decode breakdown.

## Build

```
cargo build --release
```

Windows note: developed with the GNU toolchain
(`rustup default stable-x86_64-pc-windows-gnu`); MSVC works if Build Tools
are installed.
