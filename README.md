<p align="center">
  <img src="assets/banner.svg" alt="furnace: an LLM inference engine from scratch in Rust" width="100%">
</p>

# furnace

A minimal LLM inference engine written from scratch in Rust. It loads a GGUF model file and generates text on a plain CPU — no ML frameworks, no BLAS, no `tokenizers` crate, no llama.cpp bindings. The GGUF parser, dequantization, byte-level BPE tokenizer, tensor math, transformer forward pass, KV cache, sampler, and fused quantized matmul kernels are all hand-written and individually verified against reference implementations.

```
> furnace run qwen2.5-1.5b-instruct-q4_k_m.gguf -p "Why is the sky blue?"
The sky appears blue because of a phenomenon called Rayleigh scattering...
```

**Current state:** runs Qwen2.5-1.5B-Instruct (Q4_K_M) at ~98 ms/token in ~1 GB of RAM on a Ryzen 7 5700U laptop with single-channel DDR4. Load time is 0.4 seconds because weights are never dequantized up front — they stream out of the memory-mapped file straight into fused kernels.

## Why this exists

Most of us use inference engines as black boxes. furnace is the opposite: every byte between a `.gguf` file on disk and a streamed UTF-8 token on screen is code in this repo, written to be read. It began as a learning project with one rule — no step advances until its output is pinned against a trusted reference — and ended within 2% of the machine's memory-bandwidth ceiling.

## Architecture

<p align="center">
  <img src="assets/architecture.svg" alt="Pipeline: GGUF mmap to parser, tokenizer and fused quantized kernels, transformer with KV cache, sampler, streamed text. Every stage verified against a reference." width="100%">
</p>

- **`gguf.rs`** — bounds-checked GGUF parser over `mmap`: header, metadata tree, tensor table. Everything downstream is config-driven from this metadata; moving from a 24-layer 0.5B model to a 28-layer 1.5B model with different GQA geometry required zero code changes.
- **`tokenizer.rs`** — byte-level BPE loaded entirely from GGUF metadata: GPT-2 byte-to-unicode mapping, Qwen's pre-tokenization regex, rank-driven merges, special tokens matched before BPE. Matches HuggingFace `AutoTokenizer` token-for-token across English, Hindi, code, emoji, and whitespace edge cases.
- **`quant.rs`** — block dequantization and fused dequant-matmul kernels for **Q8_0, Q5_0, Q4_K, Q6_K**. In the fused path a weight row is consumed block by block — scale read, int quants dotted against the activation slice, accumulated in f32 — so full-precision weights never exist in memory.
- **`tensor.rs`** — the six ops a transformer actually needs (matmul, rmsnorm, softmax, silu/swiglu, add, mul), with one matmul convention frozen project-wide and every op shape-asserted so mistakes fail loudly.
- **`model.rs`** — the transformer: RMSNorm, RoPE (absolute positions), grouped-query attention with a preallocated KV cache, SwiGLU FFN. Prefill and decode are distinct phases; the uncached full-recompute path survives behind `--no-cache` as a permanent reference implementation.
- **`sampler.rs`** — repeat penalty, temperature, top-k, top-p in a fixed, tested order, drawn with a hand-rolled seeded PCG32. `--temp 0` short-circuits to argmax with zero RNG draws and reproduces greedy decoding byte-for-byte.

## The journey

The project was built in ten strict milestones. No milestone was allowed to close until its checkpoint passed against an external reference.

| Milestone | What was built | Checkpoint |
|---|---|---|
| M1 | GGUF parser over mmap | 291 tensors match the official `gguf` Python reader — name, dims, dtype, offset |
| M2 | Q8_0 dequantization | Bit-exact against `gguf.quants`, verified mid-tensor, not just at offsets 0 |
| M3 | Byte-level BPE tokenizer | 18/18 diverse strings match HF `AutoTokenizer`; byte-exact round-trips |
| M4 | Tensor core (6 ops) | All ops within 1e-5 of PyTorch on seeded random inputs |
| M5 | One transformer layer | Six staged comparisons against HF forward hooks; block output at 2.8e-5 |
| M6 | Full model + greedy decoding | Last-position logits at 7e-5 vs HF; first 7 generated tokens identical |
| M7 | KV cache | Byte-identical token ids vs the uncached path; 27,000 → 372 ms/token |
| M8 | Sampling | `--temp 0` byte-identical to greedy with 0 RNG draws; seed fully determines output |
| M9 | Performance pass | 372 → 117 ms/token, measured at 1.02x of the memory-bandwidth floor |
| M10 | Fused quantized kernels, Q4_K_M | 1.5B model, 28 fused-4-bit layers, argmax and top-5 logits exact vs HF |

## Performance, honestly

All numbers from a Ryzen 7 5700U (8c/16t) with 16 GB of **single-channel** DDR4-3200, Windows, on battery-throttled base clocks. Your machine will differ; the method won't.

| Configuration | Decode (ms/token) | Resident weights |
|---|---|---|
| M6 baseline (f32, no cache) | 27,000-35,000 | 2.4 GB |
| M7 KV cache | 372 | 2.4 GB |
| M9 optimized f32 | 117 | 2.4 GB |
| M10 fused Q8_0 (0.5B) | ~35 | 531 MB |
| M10 fused Q4_K_M (1.5B) | ~98 | 1.06 GB |

Two findings from M9 worth more than the speedup itself:

1. **Decode is memory-bound.** At 117 ms/token the f32 path was moving 2.4 GB of weights per token — 20.5 GB/s against a ~21 GB/s practical single-channel ceiling. That is 1.02x of the theoretical floor.
2. **Manual SIMD was skipped, with evidence.** `target-cpu=native` measured as a wash, proving auto-vectorization was already adequate and arithmetic was not the bottleneck. No hand-written kernel makes a DIMM faster. The only remaining lever was moving fewer bytes — which is exactly what M10's fused quantized kernels do, and why they scale to all 16 threads where the f32 path saturated at 4.

Thread scaling, correctness invariants, and per-op timings are reproducible via the built-in `--timings` dashboard.

## Quick start

```bash
# 1. Build (stable Rust; on Windows the GNU toolchain avoids needing MSVC Build Tools)
cargo build --release

# 2. Get a model (any Qwen2-family GGUF in Q8_0 / Q5_0 / Q4_K_M)
curl -L -o models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf

# 3. Talk to it
furnace run models/qwen2.5-1.5b-instruct-q4_k_m.gguf -p "Explain KV caching in one paragraph"
```

## CLI

```
furnace <COMMAND>
  run          Generate text (greedy at --temp 0, sampled otherwise)
  inspect      Parse a GGUF file and dump its metadata and tensor table
  dump-tensor  Dequantize a tensor and print values as f32
  tokenize     Encode text to token ids
  detokenize   Decode token ids back to text

furnace run <MODEL> -p <PROMPT>
  -n, --max-tokens <N>       [default: 50]
      --raw                  skip the chat template
      --no-cache             uncached reference path (slow, pure greedy)
      --f32                  dequantize-up-front reference path
      --timings              per-op performance breakdown
      --temp <T>             [default: 0.7]   0 = deterministic
      --top-k <K>            [default: 40]    0 = off
      --top-p <P>            [default: 0.9]   1.0 = off
      --repeat-penalty <R>   [default: 1.0]   off by default
      --seed <SEED>          omit for clock entropy (always printed)
```

Every default is honest: `--temp 0` is exactly greedy, the repeat penalty is off so that invariant holds, and the seed is always printed so any run can be reproduced.

## Verification philosophy

The rule that shaped the whole project: **a stage that is not pinned against a reference is assumed broken.** Concretely:

- Reference scripts (in `scripts/`) compare against the `gguf` package, HuggingFace transformers (loaded in explicit f32 — bf16 references produce phantom diffs), and PyTorch ops.
- The M5 layer forward was verified in six stages against HF forward hooks — embeddings, post-norm, Q/K/V, post-RoPE, attention probabilities, block output — so a failure names its own culprit.
- Refactors that only reorganize computation (KV cache, sampling short-circuit) are held to **byte-identical** output, not tolerances.
- The staged process caught two bugs in the *references* themselves, including a silent bf16 downcast fingerprinted by the magnitude of its error.
- Every optimization in M9 landed with a same-session before/after measurement and a correctness re-check, or it was reverted.

46/46 tests. The dependency list for the core engine: `memmap2`, `half`, `anyhow`/`thiserror`, `clap`, `fancy-regex` (Qwen's pre-tokenizer needs lookahead), `rayon`. That's it.

## Supported today / not here yet

Runs Qwen2-architecture GGUF models in F32, Q8_0, Q5_0, Q4_K, and Q6_K, CPU-only, one sequence at a time.

Deliberately out of scope, at least for now: GPU backends, batching, beam search, Llama/Mistral architecture variants, speculative decoding, and other quant formats (Q4_0, IQ-series). The architecture-specific surface is small — config, RoPE conventions, tensor naming — so Llama support is the natural next step.

## License

MIT. Read it, fork it, break it, learn from it — that's what it's for.
