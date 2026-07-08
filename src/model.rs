// Qwen2 transformer block. M5 scope: config, weight loading, embedding, and
// a single block's forward pass (no KV cache; full recompute).
//
// Dataflow for seq = 5 (Qwen2.5-0.5B: hidden 896, 14 Q heads, 2 KV heads,
// head_dim 64, ffn 4864):
//
//   tokens [5] -> embed lookup (token_embd [151936,896])   x     [5, 896]
//   x  -> rmsnorm(attn_norm)                               xn    [5, 896]
//   xn -> @ Wq^T + bq                                      q     [5, 896]
//         @ Wk^T + bk                                      k     [5, 128]
//         @ Wv^T + bv                                      v     [5, 128]
//   q,k -> rope(theta 1e6, pos 0..4, half-split pairing)   shapes unchanged
//   q,k -> scores/8, causal mask, softmax                  probs [14, 5, 5]
//   probs,v -> weighted sum per head (Q head h uses KV h/7) ctx  [5, 896]
//   ctx -> @ Wo^T (no bias)                                attn  [5, 896]
//   x + attn                                               h1    [5, 896]
//   h1 -> rmsnorm(ffn_norm)                                hn    [5, 896]
//   hn -> @ Wg^T -> gate [5,4864]; @ Wu^T -> up [5,4864]
//   silu(gate) * up -> @ Wd^T                              ffn   [5, 896]
//   h1 + ffn                                               out   [5, 896]

use crate::gguf::{GgmlType, GgufFile};
use crate::tensor::{self, Tensor};
use anyhow::{ensure, Result};

#[derive(Debug, Clone)]
pub struct Config {
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub hidden: usize,
    pub ffn: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
    pub vocab: usize,
    pub context_length: usize,
}

impl Config {
    pub fn from_gguf(file: &GgufFile) -> Result<Config> {
        let arch = file.meta_str("general.architecture")?;
        ensure!(arch == "qwen2", "architecture '{}' not supported", arch);

        let n_heads = file.meta_u32("qwen2.attention.head_count")? as usize;
        let hidden = file.meta_u32("qwen2.embedding_length")? as usize;
        ensure!(hidden % n_heads == 0, "hidden {} not divisible by {} heads", hidden, n_heads);
        Ok(Config {
            n_layers: file.meta_u32("qwen2.block_count")? as usize,
            n_heads,
            n_kv_heads: file.meta_u32("qwen2.attention.head_count_kv")? as usize,
            hidden,
            ffn: file.meta_u32("qwen2.feed_forward_length")? as usize,
            head_dim: hidden / n_heads,
            rope_theta: file.meta_f32("qwen2.rope.freq_base")?,
            rms_eps: file.meta_f32("qwen2.attention.layer_norm_rms_epsilon")?,
            vocab: file.meta_array("tokenizer.ggml.tokens")?.len(),
            context_length: file.meta_u32("qwen2.context_length")? as usize,
        })
    }
}

/// A matmul weight: either owned dequantized f32 (reference path, and any
/// dtype without a fused kernel) or raw quantized bytes borrowed straight
/// from the mmap (zero-copy: resident memory is the mapped file pages).
pub enum Weight<'a> {
    F32(Tensor),
    Quant {
        raw: &'a [u8],
        dtype: GgmlType,
        out: usize,
        inn: usize,
    },
}

impl<'a> Weight<'a> {
    pub fn out_dim(&self) -> usize {
        match self {
            Weight::F32(t) => t.shape[0],
            Weight::Quant { out, .. } => *out,
        }
    }

    pub fn in_dim(&self) -> usize {
        match self {
            Weight::F32(t) => t.shape[1],
            Weight::Quant { inn, .. } => *inn,
        }
    }

    /// Bytes this weight keeps resident (f32 storage or quantized bytes).
    pub fn bytes(&self) -> usize {
        match self {
            Weight::F32(t) => t.numel() * 4,
            Weight::Quant { raw, .. } => raw.len(),
        }
    }

    pub fn matmul(&self, x: &Tensor) -> Tensor {
        match self {
            Weight::F32(t) => tensor::matmul(x, t),
            Weight::Quant { raw, dtype, out, inn } => match dtype {
                GgmlType::Q8_0 => crate::quant::matmul_q8_0(x, raw, *out, *inn),
                other => crate::quant::matmul_blocks(x, raw, *out, *inn, *other),
            },
        }
    }
}

/// Load a 2-D weight: quantized dtypes with a fused kernel stay as borrowed
/// bytes; everything else (including F32 tensors and dtypes we can only
/// dequantize) becomes owned f32. use_f32 forces the dequantized path.
pub fn load_weight<'a>(file: &'a GgufFile, name: &str, use_f32: bool) -> Result<Weight<'a>> {
    let (info, raw) = file.tensor(name)?;
    let has_fused_kernel = info.dtype == GgmlType::Q8_0
        || crate::quant::block_dequant(info.dtype).is_some();
    if use_f32 || !has_fused_kernel {
        return Ok(Weight::F32(Tensor::from_gguf(file, name)?));
    }
    ensure!(info.dims.len() == 2, "weight '{}' is not 2-D: {:?}", name, info.dims);
    Ok(Weight::Quant {
        raw,
        dtype: info.dtype,
        out: info.dims[1] as usize,
        inn: info.dims[0] as usize,
    })
}

/// One transformer block's weights. Matmul weights may be quantized; norms
/// and biases are always small owned f32 tensors.
pub struct Layer<'a> {
    pub attn_norm: Tensor, // [hidden]
    pub wq: Weight<'a>,    // [n_heads*head_dim, hidden]
    pub bq: Tensor,        // [n_heads*head_dim]
    pub wk: Weight<'a>,    // [n_kv_heads*head_dim, hidden]
    pub bk: Tensor,
    pub wv: Weight<'a>,
    pub bv: Tensor,
    pub wo: Weight<'a>,    // [hidden, n_heads*head_dim], no bias
    pub ffn_norm: Tensor,  // [hidden]
    pub w_gate: Weight<'a>, // [ffn, hidden]
    pub w_up: Weight<'a>,   // [ffn, hidden]
    pub w_down: Weight<'a>, // [hidden, ffn]
}

impl<'a> Layer<'a> {
    pub fn load(file: &'a GgufFile, idx: usize, use_f32: bool) -> Result<Layer<'a>> {
        let t = |suffix: &str| Tensor::from_gguf(file, &format!("blk.{}.{}", idx, suffix));
        let w = |suffix: &str| load_weight(file, &format!("blk.{}.{}", idx, suffix), use_f32);
        Ok(Layer {
            attn_norm: t("attn_norm.weight")?,
            wq: w("attn_q.weight")?,
            bq: t("attn_q.bias")?,
            wk: w("attn_k.weight")?,
            bk: t("attn_k.bias")?,
            wv: w("attn_v.weight")?,
            bv: t("attn_v.bias")?,
            wo: w("attn_output.weight")?,
            ffn_norm: t("ffn_norm.weight")?,
            w_gate: w("ffn_gate.weight")?,
            w_up: w("ffn_up.weight")?,
            w_down: w("ffn_down.weight")?,
        })
    }
}

/// x [seq, in] @ w [out, in] -> [seq, out], plus optional bias [out]
/// broadcast over rows. Dispatches on the weight storage (f32 or fused
/// quantized kernel).
pub fn linear(x: &Tensor, w: &Weight, bias: Option<&Tensor>) -> Tensor {
    let mut out = w.matmul(x);
    if let Some(b) = bias {
        let cols = out.shape[1];
        assert_eq!(
            b.shape,
            vec![cols],
            "linear: bias shape {:?} does not match output width {}",
            b.shape, cols
        );
        for row in out.data.chunks_exact_mut(cols) {
            for (o, &bv) in row.iter_mut().zip(&b.data) {
                *o += bv;
            }
        }
    }
    out
}

/// Rotary position embedding, half-split (NeoX) pairing as used by HF Qwen2:
/// within each head, dimension i pairs with i + head_dim/2 and the pair is
/// rotated by pos * theta^(-2i/head_dim). Row s of x is at ABSOLUTE position
/// start_pos + s: with a KV cache, a decoded token is row 0 of its batch but
/// is not at position 0. Applied in place to Q or K; V is never rotated.
pub fn rope(x: &mut Tensor, n_heads: usize, head_dim: usize, theta: f32, start_pos: usize) {
    let seq = x.shape[0];
    assert_eq!(
        x.shape[1],
        n_heads * head_dim,
        "rope: row width {} != {} heads x {}",
        x.shape[1], n_heads, head_dim
    );
    let _t = crate::perf::time(&crate::perf::ROPE);
    let half = head_dim / 2;
    for s in 0..seq {
        let row = &mut x.data[s * n_heads * head_dim..(s + 1) * n_heads * head_dim];
        for h in 0..n_heads {
            let head = &mut row[h * head_dim..(h + 1) * head_dim];
            for i in 0..half {
                let freq = theta.powf(-2.0 * i as f32 / head_dim as f32);
                let angle = (start_pos + s) as f32 * freq;
                let (sin, cos) = angle.sin_cos();
                let (a, b) = (head[i], head[i + half]);
                head[i] = a * cos - b * sin;
                head[i + half] = a * sin + b * cos;
            }
        }
    }
}

/// Causal attention probabilities with grouped-query head mapping.
/// q [seq, n_heads*head_dim], k [seq, n_kv_heads*head_dim] (both post-RoPE)
/// -> [n_heads, seq, seq] post-softmax. Q head h reads KV head h/group where
/// group = n_heads / n_kv_heads. score[s][t] = -inf for t > s (causal),
/// finite scores scaled by 1/sqrt(head_dim).
pub fn attention_scores(
    q: &Tensor,
    k: &Tensor,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Tensor {
    let seq = q.shape[0];
    assert_eq!(q.shape[1], n_heads * head_dim, "attention: bad q width {:?}", q.shape);
    assert_eq!(k.shape[1], n_kv_heads * head_dim, "attention: bad k width {:?}", k.shape);
    assert_eq!(k.shape[0], seq, "attention: q seq {} != k seq {}", seq, k.shape[0]);
    assert_eq!(n_heads % n_kv_heads, 0, "attention: {} q heads not divisible by {} kv heads", n_heads, n_kv_heads);
    let group = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut probs = Tensor::zeros(vec![n_heads, seq, seq]);
    for h in 0..n_heads {
        let kv = h / group;
        for s in 0..seq {
            let q_head = &q.data[s * q.shape[1] + h * head_dim..][..head_dim];
            let row = &mut probs.data[(h * seq + s) * seq..][..seq];
            for t in 0..seq {
                row[t] = if t <= s {
                    let k_head = &k.data[t * k.shape[1] + kv * head_dim..][..head_dim];
                    let dot: f32 = q_head.iter().zip(k_head).map(|(a, b)| a * b).sum();
                    dot * scale
                } else {
                    f32::NEG_INFINITY
                };
            }
        }
    }
    tensor::softmax(&mut probs);
    probs
}

/// Weighted sum of V rows: probs [n_heads, seq, seq] x v [seq, n_kv_heads*
/// head_dim] -> [seq, n_heads*head_dim] with heads concatenated per row.
pub fn attention_apply(
    probs: &Tensor,
    v: &Tensor,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Tensor {
    let seq = v.shape[0];
    assert_eq!(probs.shape, vec![n_heads, seq, seq], "attention_apply: bad probs shape {:?}", probs.shape);
    assert_eq!(v.shape[1], n_kv_heads * head_dim, "attention_apply: bad v width {:?}", v.shape);
    let group = n_heads / n_kv_heads;

    let mut out = Tensor::zeros(vec![seq, n_heads * head_dim]);
    for h in 0..n_heads {
        let kv = h / group;
        for s in 0..seq {
            let row = &probs.data[(h * seq + s) * seq..][..seq];
            let out_head = &mut out.data[s * n_heads * head_dim + h * head_dim..][..head_dim];
            for t in 0..seq {
                let p = row[t];
                let v_head = &v.data[t * v.shape[1] + kv * head_dim..][..head_dim];
                for d in 0..head_dim {
                    out_head[d] += p * v_head[d];
                }
            }
        }
    }
    out
}

/// One full transformer block: x [seq, hidden] -> [seq, hidden].
/// See the dataflow diagram in the module header.
pub fn layer_forward(x: &Tensor, layer: &Layer, config: &Config) -> Tensor {
    let (nh, nkv, hd) = (config.n_heads, config.n_kv_heads, config.head_dim);

    // attention half
    let xn = tensor::rmsnorm(x, &layer.attn_norm, config.rms_eps);
    let mut q = linear(&xn, &layer.wq, Some(&layer.bq));
    let mut k = linear(&xn, &layer.wk, Some(&layer.bk));
    let v = linear(&xn, &layer.wv, Some(&layer.bv));
    rope(&mut q, nh, hd, config.rope_theta, 0);
    rope(&mut k, nkv, hd, config.rope_theta, 0);
    let probs = attention_scores(&q, &k, nh, nkv, hd);
    let ctx = attention_apply(&probs, &v, nh, nkv, hd);
    let attn = linear(&ctx, &layer.wo, None);
    let h1 = tensor::add(x, &attn);

    // ffn half
    let hn = tensor::rmsnorm(&h1, &layer.ffn_norm, config.rms_eps);
    let gate = linear(&hn, &layer.w_gate, None);
    let up = linear(&hn, &layer.w_up, None);
    let ffn = linear(&tensor::swiglu(&gate, &up), &layer.w_down, None);
    tensor::add(&h1, &ffn)
}

/// Per-layer K/V storage for autoregressive decoding. K rows are stored
/// post-RoPE (a position's rotation never changes); V rows as-is. Buffers
/// are pre-allocated to max_ctx positions; len counts positions filled and
/// advances once per forward pass, after every layer has appended.
pub struct KvCache {
    k: Vec<Vec<f32>>, // [n_layers][max_ctx * kv_dim]
    v: Vec<Vec<f32>>,
    pub len: usize,
    pub max_ctx: usize,
    kv_dim: usize,
}

impl KvCache {
    pub fn new(config: &Config, max_ctx: usize) -> KvCache {
        let kv_dim = config.n_kv_heads * config.head_dim;
        KvCache {
            k: vec![vec![0.0; max_ctx * kv_dim]; config.n_layers],
            v: vec![vec![0.0; max_ctx * kv_dim]; config.n_layers],
            len: 0,
            max_ctx,
            kv_dim,
        }
    }

    pub fn bytes(&self) -> usize {
        2 * self.k.len() * self.max_ctx * self.kv_dim * 4
    }

    /// Write new K/V rows for one layer at positions start_pos..start_pos+n.
    fn append(&mut self, layer_idx: usize, start_pos: usize, k_rows: &[f32], v_rows: &[f32]) {
        let n = k_rows.len() / self.kv_dim;
        assert!(
            start_pos + n <= self.max_ctx,
            "kv cache overflow: {} + {} > max_ctx {}",
            start_pos, n, self.max_ctx
        );
        let at = start_pos * self.kv_dim;
        self.k[layer_idx][at..at + k_rows.len()].copy_from_slice(k_rows);
        self.v[layer_idx][at..at + v_rows.len()].copy_from_slice(v_rows);
    }

    /// Cached K and V for one layer, valid through position `total`.
    fn layer(&self, layer_idx: usize, total: usize) -> (&[f32], &[f32]) {
        let end = total * self.kv_dim;
        (&self.k[layer_idx][..end], &self.v[layer_idx][..end])
    }
}

/// Attention probabilities for new queries against the cached K range.
/// q [n_new, n_heads*head_dim] at absolute positions start_pos..start_pos+
/// n_new; k_cache holds `total` rows of kv_dim. -> [n_heads, n_new, total].
/// Query s attends to t <= start_pos + s; in decode (n_new = 1) every cached
/// position satisfies this -- the cache holds only the past, so attending to
/// all of it is causal by construction.
fn attention_scores_cached(
    q: &Tensor,
    k_cache: &[f32],
    total: usize,
    start_pos: usize,
    config: &Config,
) -> Tensor {
    let (nh, nkv, hd) = (config.n_heads, config.n_kv_heads, config.head_dim);
    let n_new = q.shape[0];
    let kv_dim = nkv * hd;
    assert_eq!(q.shape[1], nh * hd, "attention: bad q width {:?}", q.shape);
    assert_eq!(k_cache.len(), total * kv_dim, "attention: bad k_cache length");
    let group = nh / nkv;
    let scale = 1.0 / (hd as f32).sqrt();

    let mut probs = Tensor::zeros(vec![nh, n_new, total]);
    let t_scores = crate::perf::time(&crate::perf::ATTN_DOT);
    for h in 0..nh {
        let kv = h / group;
        for s in 0..n_new {
            let q_head = &q.data[s * q.shape[1] + h * hd..][..hd];
            let row = &mut probs.data[(h * n_new + s) * total..][..total];
            for t in 0..total {
                row[t] = if t <= start_pos + s {
                    let k_head = &k_cache[t * kv_dim + kv * hd..][..hd];
                    let dot: f32 = q_head.iter().zip(k_head).map(|(a, b)| a * b).sum();
                    dot * scale
                } else {
                    f32::NEG_INFINITY
                };
            }
        }
    }
    drop(t_scores); // softmax accounts for itself
    tensor::softmax(&mut probs);
    probs
}

/// probs [n_heads, n_new, total] x cached V -> [n_new, n_heads*head_dim].
fn attention_apply_cached(
    probs: &Tensor,
    v_cache: &[f32],
    config: &Config,
) -> Tensor {
    let (nh, nkv, hd) = (config.n_heads, config.n_kv_heads, config.head_dim);
    let (n_new, total) = (probs.shape[1], probs.shape[2]);
    let kv_dim = nkv * hd;
    assert_eq!(v_cache.len(), total * kv_dim, "attention: bad v_cache length");
    let group = nh / nkv;

    let _t = crate::perf::time(&crate::perf::ATTN_APPLY);
    let mut out = Tensor::zeros(vec![n_new, nh * hd]);
    for h in 0..nh {
        let kv = h / group;
        for s in 0..n_new {
            let row = &probs.data[(h * n_new + s) * total..][..total];
            let out_head = &mut out.data[s * nh * hd + h * hd..][..hd];
            for t in 0..total {
                let p = row[t];
                let v_head = &v_cache[t * kv_dim + kv * hd..][..hd];
                for d in 0..hd {
                    out_head[d] += p * v_head[d];
                }
            }
        }
    }
    out
}

/// One block over NEW tokens only, reading/writing the KV cache.
/// x [n_new, hidden] are the hidden states of tokens at absolute positions
/// start_pos..start_pos+n_new. Prefill is n_new = prompt length with an
/// empty cache; decode is n_new = 1.
pub fn layer_forward_cached(
    x: &Tensor,
    layer: &Layer,
    config: &Config,
    cache: &mut KvCache,
    layer_idx: usize,
    start_pos: usize,
) -> Tensor {
    let (nh, nkv, hd) = (config.n_heads, config.n_kv_heads, config.head_dim);
    let n_new = x.shape[0];

    // attention half
    let xn = tensor::rmsnorm(x, &layer.attn_norm, config.rms_eps);
    let mut q = linear(&xn, &layer.wq, Some(&layer.bq));
    let mut k = linear(&xn, &layer.wk, Some(&layer.bk));
    let v = linear(&xn, &layer.wv, Some(&layer.bv));
    rope(&mut q, nh, hd, config.rope_theta, start_pos);
    rope(&mut k, nkv, hd, config.rope_theta, start_pos);

    cache.append(layer_idx, start_pos, &k.data, &v.data);
    let total = start_pos + n_new;
    let (k_cache, v_cache) = cache.layer(layer_idx, total);
    let probs = attention_scores_cached(&q, k_cache, total, start_pos, config);
    let ctx = attention_apply_cached(&probs, v_cache, config);

    let attn = linear(&ctx, &layer.wo, None);
    let h1 = tensor::add(x, &attn);

    // ffn half (identical to the uncached path)
    let hn = tensor::rmsnorm(&h1, &layer.ffn_norm, config.rms_eps);
    let gate = linear(&hn, &layer.w_gate, None);
    let up = linear(&hn, &layer.w_up, None);
    let ffn = linear(&tensor::swiglu(&gate, &up), &layer.w_down, None);
    tensor::add(&h1, &ffn)
}

/// Token embedding lookup. For quantized embeddings only the touched rows
/// are dequantized (a row is a whole number of blocks, so it slices clean).
pub fn embed(embedding: &Weight, tokens: &[u32]) -> Tensor {
    let _t = crate::perf::time(&crate::perf::EMBED);
    let hidden = embedding.in_dim();
    let vocab = embedding.out_dim();
    let mut out = Tensor::zeros(vec![tokens.len(), hidden]);
    for (s, &tok) in tokens.iter().enumerate() {
        let t = tok as usize;
        assert!(t < vocab, "token id {} out of range (vocab {})", t, vocab);
        let dst = &mut out.data[s * hidden..(s + 1) * hidden];
        match embedding {
            Weight::F32(e) => dst.copy_from_slice(&e.data[t * hidden..(t + 1) * hidden]),
            Weight::Quant { raw, dtype, .. } => {
                let (block_elems, block_bytes) = dtype
                    .block_layout()
                    .expect("quant weight with unknown block layout");
                let row_bytes = hidden / block_elems * block_bytes;
                let row = &raw[t * row_bytes..(t + 1) * row_bytes];
                let values = crate::quant::dequantize(*dtype, row, hidden)
                    .expect("embedding row dequantization failed");
                dst.copy_from_slice(&values);
            }
        }
    }
    out
}

/// The whole network: embedding, all blocks, final norm, lm_head.
/// Weights stay quantized (borrowed from the mmap) unless use_f32 asks for
/// the dequantize-everything reference path.
pub struct Model<'a> {
    pub config: Config,
    pub embedding: Weight<'a>, // [vocab, hidden]
    pub layers: Vec<Layer<'a>>,
    pub output_norm: Tensor,   // [hidden]
    pub lm_head: Weight<'a>,   // [vocab, hidden]
}

impl<'a> Model<'a> {
    pub fn load(file: &'a GgufFile, use_f32: bool) -> Result<Model<'a>> {
        let config = Config::from_gguf(file)?;
        let embedding = load_weight(file, "token_embd.weight", use_f32)?;
        ensure!(
            embedding.out_dim() == config.vocab && embedding.in_dim() == config.hidden,
            "token_embd is [{}, {}], config says [{}, {}]",
            embedding.out_dim(), embedding.in_dim(), config.vocab, config.hidden
        );
        let layers = (0..config.n_layers)
            .map(|i| Layer::load(file, i, use_f32))
            .collect::<Result<Vec<_>>>()?;
        for (i, l) in layers.iter().enumerate() {
            ensure!(
                l.w_gate.out_dim() == config.ffn && l.w_gate.in_dim() == config.hidden,
                "blk.{} ffn_gate is [{}, {}], expected [{}, {}]",
                i, l.w_gate.out_dim(), l.w_gate.in_dim(), config.ffn, config.hidden
            );
        }
        let output_norm = Tensor::from_gguf(file, "output_norm.weight")?;
        // tied-embedding models (e.g. Qwen2.5-1.5B) omit output.weight from
        // the GGUF; the lm_head is then the embedding matrix itself
        let lm_head = if file.tensors.iter().any(|t| t.name == "output.weight") {
            load_weight(file, "output.weight", use_f32)?
        } else {
            load_weight(file, "token_embd.weight", use_f32)?
        };
        ensure!(
            lm_head.out_dim() == config.vocab && lm_head.in_dim() == config.hidden,
            "lm_head is [{}, {}], config says [{}, {}]",
            lm_head.out_dim(), lm_head.in_dim(), config.vocab, config.hidden
        );
        Ok(Model { config, embedding, layers, output_norm, lm_head })
    }

    /// Bytes the weights keep resident (quantized bytes or f32 storage).
    pub fn param_bytes(&self) -> usize {
        let layer_bytes: usize = self.layers.iter().map(|l| {
            l.wq.bytes() + l.wk.bytes() + l.wv.bytes() + l.wo.bytes()
                + l.w_gate.bytes() + l.w_up.bytes() + l.w_down.bytes()
                + (l.attn_norm.numel() + l.bq.numel() + l.bk.numel()
                    + l.bv.numel() + l.ffn_norm.numel()) * 4
        }).sum();
        layer_bytes + self.embedding.bytes() + self.output_norm.numel() * 4 + self.lm_head.bytes()
    }

    /// Run the full network and return logits for the LAST position only:
    /// [1, vocab]. Greedy decoding never needs the other rows, and the
    /// lm_head matmul is the single most expensive op in the model.
    /// Uncached reference path: recomputes everything, every call.
    pub fn forward_last(&self, tokens: &[u32]) -> Tensor {
        let mut x = embed(&self.embedding, tokens);
        for layer in &self.layers {
            x = layer_forward(&x, layer, &self.config);
        }
        self.last_row_logits(&x)
    }

    /// Cached forward over NEW tokens only. Pass the whole prompt for
    /// prefill, then one token per decode step. Advances cache.len.
    pub fn forward_cached(&self, tokens: &[u32], cache: &mut KvCache) -> Tensor {
        let start = cache.len;
        assert!(
            start + tokens.len() <= cache.max_ctx,
            "context overflow: {} + {} > max_ctx {}",
            start, tokens.len(), cache.max_ctx
        );
        let mut x = embed(&self.embedding, tokens);
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer_forward_cached(&x, layer, &self.config, cache, i, start);
        }
        cache.len += tokens.len();
        self.last_row_logits(&x)
    }

    fn last_row_logits(&self, x: &Tensor) -> Tensor {
        let xn = tensor::rmsnorm(x, &self.output_norm, self.config.rms_eps);
        let hidden = self.config.hidden;
        let last_row = xn.data[(xn.shape[0] - 1) * hidden..].to_vec();
        let last = Tensor::from_vec(last_row, vec![1, hidden]);
        linear(&last, &self.lm_head, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(got: &[f32], want: &[f32], tol: f32) {
        assert_eq!(got.len(), want.len());
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() <= tol, "element {}: got {}, want {}", i, g, w);
        }
    }

    #[test]
    fn rope_position_zero_is_identity() {
        // row 0 is position 0: all angles are 0, rotation is identity
        let mut x = Tensor::from_vec(vec![1., 2., 3., 4.], vec![1, 4]);
        rope(&mut x, 1, 4, 1e6, 0);
        assert_eq!(x.data, vec![1., 2., 3., 4.]);
    }

    #[test]
    fn rope_known_vector_at_position_one() {
        // head_dim 4, theta 4: pair (x0,x2) rotates by pos*4^0 = 1 rad,
        // pair (x1,x3) by pos*4^(-1/2) = 0.5 rad. At pos 1, x = [1,2,3,4]:
        //   x0' = 1*cos1 - 3*sin1 = 0.5403023 - 2.5244131 = -1.9841108
        //   x1' = 2*cos.5 - 4*sin.5 = 1.7551652 - 1.9177020 = -0.1625368
        //   x2' = 1*sin1 + 3*cos1 = 0.8414710 + 1.6209068 =  2.4623778
        //   x3' = 2*sin.5 + 4*cos.5 = 0.9588511 + 3.5103304 = 4.4691815
        let mut x = Tensor::from_vec(vec![9., 9., 9., 9., 1., 2., 3., 4.], vec![2, 4]);
        rope(&mut x, 1, 4, 4.0, 0);
        assert_close(
            &x.data[4..],
            &[-1.9841108, -0.1625368, 2.4623778, 4.4691815],
            1e-5,
        );
    }

    #[test]
    fn causal_mask_zeroes_future_tokens() {
        let q = Tensor::from_vec(vec![1., 0., 0., 1., 1., 1.], vec![3, 2]);
        let k = Tensor::from_vec(vec![1., 1., 0., 1., 1., 0.], vec![3, 2]);
        let probs = attention_scores(&q, &k, 1, 1, 2);
        assert_eq!(probs.shape, vec![1, 3, 3]);
        // token 0 may only see itself: probability exactly 1, futures exactly 0
        assert_eq!(probs.data[0], 1.0);
        assert_eq!(probs.data[1], 0.0);
        assert_eq!(probs.data[2], 0.0);
        // token 1 must not see token 2
        assert_eq!(probs.data[5], 0.0);
        // every row sums to 1
        for row in probs.data.chunks_exact(3) {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "row sums to {}", sum);
        }
    }

    /// Deterministic pseudo-random values, same every run.
    fn fill(n: usize, salt: f32) -> Vec<f32> {
        (0..n).map(|i| ((i as f32 * 0.37 + salt).sin()) * 0.5).collect()
    }

    fn tiny_config() -> Config {
        Config {
            n_layers: 1,
            n_heads: 2,
            n_kv_heads: 1,
            hidden: 8,
            ffn: 16,
            head_dim: 4,
            rope_theta: 100.0,
            rms_eps: 1e-6,
            vocab: 1,
            context_length: 32,
        }
    }

    fn tiny_layer(c: &Config) -> Layer<'static> {
        let (h, f, qd, kd) = (c.hidden, c.ffn, c.n_heads * c.head_dim, c.n_kv_heads * c.head_dim);
        Layer {
            attn_norm: Tensor::from_vec(fill(h, 1.0), vec![h]),
            wq: Weight::F32(Tensor::from_vec(fill(qd * h, 2.0), vec![qd, h])),
            bq: Tensor::from_vec(fill(qd, 3.0), vec![qd]),
            wk: Weight::F32(Tensor::from_vec(fill(kd * h, 4.0), vec![kd, h])),
            bk: Tensor::from_vec(fill(kd, 5.0), vec![kd]),
            wv: Weight::F32(Tensor::from_vec(fill(kd * h, 6.0), vec![kd, h])),
            bv: Tensor::from_vec(fill(kd, 7.0), vec![kd]),
            wo: Weight::F32(Tensor::from_vec(fill(h * qd, 8.0), vec![h, qd])),
            ffn_norm: Tensor::from_vec(fill(h, 9.0), vec![h]),
            w_gate: Weight::F32(Tensor::from_vec(fill(f * h, 10.0), vec![f, h])),
            w_up: Weight::F32(Tensor::from_vec(fill(f * h, 11.0), vec![f, h])),
            w_down: Weight::F32(Tensor::from_vec(fill(h * f, 12.0), vec![h, f])),
        }
    }

    #[test]
    fn cached_path_is_bit_identical_to_uncached() {
        // 3-token prefill + 2 single-token decode steps must equal one
        // 5-token uncached forward EXACTLY: both paths execute the same
        // dot products in the same order, so even f32 rounding agrees.
        let config = tiny_config();
        let layer = tiny_layer(&config);
        let x = Tensor::from_vec(fill(5 * config.hidden, 20.0), vec![5, config.hidden]);

        let uncached = layer_forward(&x, &layer, &config);

        let mut cache = KvCache::new(&config, 32);
        let prefill_in = Tensor::from_vec(x.data[..3 * config.hidden].to_vec(), vec![3, config.hidden]);
        let prefill_out = layer_forward_cached(&prefill_in, &layer, &config, &mut cache, 0, 0);
        cache.len = 3;
        // prefill rows must already match rows 0..3
        assert_eq!(prefill_out.data, uncached.data[..3 * config.hidden]);

        for s in 3..5 {
            let row = Tensor::from_vec(
                x.data[s * config.hidden..(s + 1) * config.hidden].to_vec(),
                vec![1, config.hidden],
            );
            let start = cache.len;
            let out = layer_forward_cached(&row, &layer, &config, &mut cache, 0, start);
            cache.len += 1;
            assert_eq!(
                out.data,
                uncached.data[s * config.hidden..(s + 1) * config.hidden],
                "decode step for position {} diverged from uncached",
                s
            );
        }
    }

    #[test]
    fn gqa_maps_q_heads_to_shared_kv_heads() {
        // 4 Q heads, 2 KV heads, head_dim 2, seq 1: probs are trivially 1,
        // so out head h must equal V of KV head h/2:
        // heads 0,1 -> kv0 = [10,20]; heads 2,3 -> kv1 = [30,40]
        let q = Tensor::from_vec(vec![1.; 8], vec![1, 8]);
        let k = Tensor::from_vec(vec![1.; 4], vec![1, 4]);
        let v = Tensor::from_vec(vec![10., 20., 30., 40.], vec![1, 4]);
        let probs = attention_scores(&q, &k, 4, 2, 2);
        let out = attention_apply(&probs, &v, 4, 2, 2);
        assert_eq!(out.shape, vec![1, 8]);
        assert_eq!(out.data, vec![10., 20., 10., 20., 30., 40., 30., 40.]);
    }
}
