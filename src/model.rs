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

use crate::gguf::GgufFile;
use crate::tensor::{self, Tensor};
use anyhow::{ensure, Result};

#[derive(Debug, Clone)]
// n_layers and vocab are read starting with the M6 full forward pass
#[allow(dead_code)]
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
        })
    }
}

/// One transformer block's weights, dequantized to f32.
pub struct Layer {
    pub attn_norm: Tensor, // [hidden]
    pub wq: Tensor,        // [n_heads*head_dim, hidden]
    pub bq: Tensor,        // [n_heads*head_dim]
    pub wk: Tensor,        // [n_kv_heads*head_dim, hidden]
    pub bk: Tensor,
    pub wv: Tensor,
    pub bv: Tensor,
    pub wo: Tensor,        // [hidden, n_heads*head_dim], no bias
    pub ffn_norm: Tensor,  // [hidden]
    pub w_gate: Tensor,    // [ffn, hidden]
    pub w_up: Tensor,      // [ffn, hidden]
    pub w_down: Tensor,    // [hidden, ffn]
}

impl Layer {
    pub fn load(file: &GgufFile, idx: usize) -> Result<Layer> {
        let t = |suffix: &str| Tensor::from_gguf(file, &format!("blk.{}.{}", idx, suffix));
        Ok(Layer {
            attn_norm: t("attn_norm.weight")?,
            wq: t("attn_q.weight")?,
            bq: t("attn_q.bias")?,
            wk: t("attn_k.weight")?,
            bk: t("attn_k.bias")?,
            wv: t("attn_v.weight")?,
            bv: t("attn_v.bias")?,
            wo: t("attn_output.weight")?,
            ffn_norm: t("ffn_norm.weight")?,
            w_gate: t("ffn_gate.weight")?,
            w_up: t("ffn_up.weight")?,
            w_down: t("ffn_down.weight")?,
        })
    }
}

/// x [seq, in] @ w [out, in] -> [seq, out], plus optional bias [out]
/// broadcast over rows.
pub fn linear(x: &Tensor, w: &Tensor, bias: Option<&Tensor>) -> Tensor {
    let mut out = tensor::matmul(x, w);
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
/// rotated by pos * theta^(-2i/head_dim). Row s of x is at position s.
/// Applied in place to Q or K; V is never rotated.
pub fn rope(x: &mut Tensor, n_heads: usize, head_dim: usize, theta: f32) {
    let seq = x.shape[0];
    assert_eq!(
        x.shape[1],
        n_heads * head_dim,
        "rope: row width {} != {} heads x {}",
        x.shape[1], n_heads, head_dim
    );
    let half = head_dim / 2;
    for s in 0..seq {
        let row = &mut x.data[s * n_heads * head_dim..(s + 1) * n_heads * head_dim];
        for h in 0..n_heads {
            let head = &mut row[h * head_dim..(h + 1) * head_dim];
            for i in 0..half {
                let freq = theta.powf(-2.0 * i as f32 / head_dim as f32);
                let angle = s as f32 * freq;
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
    rope(&mut q, nh, hd, config.rope_theta);
    rope(&mut k, nkv, hd, config.rope_theta);
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

/// Token embedding lookup: copy row t of token_embd for each token.
pub fn embed(embedding: &Tensor, tokens: &[u32]) -> Tensor {
    let hidden = embedding.shape[1];
    let vocab = embedding.shape[0];
    let mut out = Tensor::zeros(vec![tokens.len(), hidden]);
    for (s, &tok) in tokens.iter().enumerate() {
        let t = tok as usize;
        assert!(t < vocab, "token id {} out of range (vocab {})", t, vocab);
        out.data[s * hidden..(s + 1) * hidden]
            .copy_from_slice(&embedding.data[t * hidden..(t + 1) * hidden]);
    }
    out
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
        rope(&mut x, 1, 4, 1e6);
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
        rope(&mut x, 1, 4, 4.0);
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
