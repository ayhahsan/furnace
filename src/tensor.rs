// Minimal f32 tensor library. Naive implementations; correctness and clear
// layout semantics only (optimization is M9).
//
// Layout: row-major, last axis contiguous.
//
// Matmul convention, fixed for the whole project:
//   activations [seq, in] x weights [out, in] -> [seq, out]
//   C[s][o] = dot(A row s, W row o)        (i.e. C = A @ W^T)
// This matches ggml/GGUF storage: dims[0] (contiguous) = in_features, so a
// weight loads as [out, in] with every row contiguous and no transpose.

use crate::gguf::GgufFile;
use anyhow::Result;
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    // read by tests only until non-contiguous views appear
    #[allow(dead_code)]
    pub strides: Vec<usize>,
}

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

impl Tensor {
    pub fn zeros(shape: Vec<usize>) -> Tensor {
        let n = shape.iter().product();
        Tensor { data: vec![0.0; n], strides: row_major_strides(&shape), shape }
    }

    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        let n: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            n,
            "from_vec: {} values do not fill shape {:?}",
            data.len(),
            shape
        );
        Tensor { data, strides: row_major_strides(&shape), shape }
    }

    /// Load a GGUF tensor and dequantize to f32. GGUF dims are in ggml order
    /// (fastest-varying first), so the shape is the dims list reversed; the
    /// data itself is already laid out correctly and is not moved.
    #[allow(dead_code)] // first non-test caller is the M5 layer forward pass
    pub fn from_gguf(file: &GgufFile, name: &str) -> Result<Tensor> {
        let (info, raw) = file.tensor(name)?;
        let data = crate::quant::dequantize(info.dtype, raw, info.n_elements())?;
        let shape: Vec<usize> = info.dims.iter().rev().map(|&d| d as usize).collect();
        Ok(Tensor::from_vec(data, shape))
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Rows and row length when treating the tensor as [n_rows, last_dim].
    fn rows(&self) -> (usize, usize) {
        let cols = *self.shape.last().expect("op on 0-d tensor");
        (self.numel() / cols, cols)
    }
}

/// [seq, in] x [out, in] -> [seq, out]. See module header for the convention.
pub fn matmul(a: &Tensor, w: &Tensor) -> Tensor {
    assert_eq!(a.shape.len(), 2, "matmul: activation must be 2-D, got {:?}", a.shape);
    assert_eq!(w.shape.len(), 2, "matmul: weight must be 2-D, got {:?}", w.shape);
    let (seq, d_in) = (a.shape[0], a.shape[1]);
    let (d_out, w_in) = (w.shape[0], w.shape[1]);
    assert_eq!(
        d_in, w_in,
        "matmul: inner dims differ: activation {:?} vs weight {:?} (weight must be [out, in])",
        a.shape, w.shape
    );
    let _t = crate::perf::time(&crate::perf::MATMUL);

    // Parallel over output positions: each chunk of the output row is a
    // batch of independent dot products against consecutive weight rows.
    // Chunked so one rayon task amortizes scheduling over 64 dots.
    let mut out = Tensor::zeros(vec![seq, d_out]);
    for s in 0..seq {
        let a_row = &a.data[s * d_in..(s + 1) * d_in];
        out.data[s * d_out..(s + 1) * d_out]
            .par_chunks_mut(64)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                for (j, dst) in chunk.iter_mut().enumerate() {
                    let o = chunk_idx * 64 + j;
                    *dst = dot(a_row, &w.data[o * d_in..(o + 1) * d_in]);
                }
            });
    }
    out
}

/// Dot product with 8 independent accumulator lanes. A single-accumulator
/// loop is a serial float dependency chain the compiler must not reorder
/// (float adds are not associative), so it can neither pipeline nor
/// vectorize; independent lanes remove the chain at the cost of a different
/// (grouped) summation order.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let mut ca = a.chunks_exact(8);
    let mut cb = b.chunks_exact(8);
    for (x, y) in (&mut ca).zip(&mut cb) {
        for j in 0..8 {
            acc[j] += x[j] * y[j];
        }
    }
    let mut sum = acc.iter().sum::<f32>();
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        sum += x * y;
    }
    sum
}

/// Row-wise RMS normalization: y = x / sqrt(mean(x^2) + eps) * weight.
/// eps sits inside the sqrt, matching Llama/Qwen. Applied over the last dim.
pub fn rmsnorm(x: &Tensor, weight: &Tensor, eps: f32) -> Tensor {
    let (n_rows, cols) = x.rows();
    assert_eq!(
        weight.shape,
        vec![cols],
        "rmsnorm: weight shape {:?} does not match row length {}",
        weight.shape, cols
    );
    let _t = crate::perf::time(&crate::perf::RMSNORM);

    let mut out = Tensor::zeros(x.shape.clone());
    for r in 0..n_rows {
        let row = &x.data[r * cols..(r + 1) * cols];
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / cols as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();
        let out_row = &mut out.data[r * cols..(r + 1) * cols];
        for i in 0..cols {
            out_row[i] = row[i] * inv_rms * weight.data[i];
        }
    }
    out
}

/// In-place row-wise softmax over the last dim, max-subtracted for stability.
pub fn softmax(x: &mut Tensor) {
    let _t = crate::perf::time(&crate::perf::SOFTMAX);
    let (n_rows, cols) = x.rows();
    for r in 0..n_rows {
        let row = &mut x.data[r * cols..(r + 1) * cols];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
}

/// Elementwise SiLU: x * sigmoid(x).
#[allow(dead_code)] // swiglu fuses this; standalone version kept for M5 debugging
pub fn silu(x: &Tensor) -> Tensor {
    let data = x.data.iter().map(|&v| v / (1.0 + (-v).exp())).collect();
    Tensor::from_vec(data, x.shape.clone())
}

/// SwiGLU gating: silu(gate) * up, elementwise.
pub fn swiglu(gate: &Tensor, up: &Tensor) -> Tensor {
    assert_eq!(
        gate.shape, up.shape,
        "swiglu: gate shape {:?} vs up shape {:?}",
        gate.shape, up.shape
    );
    let _t = crate::perf::time(&crate::perf::ELEMENTWISE);
    let data = gate
        .data
        .iter()
        .zip(&up.data)
        .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
        .collect();
    Tensor::from_vec(data, gate.shape.clone())
}

pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.shape, b.shape, "add: shape {:?} vs {:?}", a.shape, b.shape);
    let _t = crate::perf::time(&crate::perf::ELEMENTWISE);
    let data = a.data.iter().zip(&b.data).map(|(&x, &y)| x + y).collect();
    Tensor::from_vec(data, a.shape.clone())
}

pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.shape, b.shape, "mul: shape {:?} vs {:?}", a.shape, b.shape);
    let data = a.data.iter().zip(&b.data).map(|(&x, &y)| x * y).collect();
    Tensor::from_vec(data, a.shape.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn assert_close(got: &[f32], want: &[f32], tol: f32) {
        assert_eq!(got.len(), want.len());
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= tol,
                "element {}: got {}, want {} (tol {})",
                i, g, w, tol
            );
        }
    }

    #[test]
    fn matmul_hand_computed() {
        // A = [[1,2,3],   W = [[1,0,1],     C[s][o] = dot(A row s, W row o)
        //      [4,5,6]]        [0,1,0]]
        // C[0][0] = 1*1 + 2*0 + 3*1 = 4     C[0][1] = 1*0 + 2*1 + 3*0 = 2
        // C[1][0] = 4*1 + 5*0 + 6*1 = 10    C[1][1] = 4*0 + 5*1 + 6*0 = 5
        let a = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
        let w = Tensor::from_vec(vec![1., 0., 1., 0., 1., 0.], vec![2, 3]);
        let c = matmul(&a, &w);
        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.data, vec![4., 2., 10., 5.]);
    }

    #[test]
    #[should_panic(expected = "matmul: inner dims differ")]
    fn matmul_rejects_mismatched_inner_dim() {
        let a = Tensor::zeros(vec![2, 3]);
        let w = Tensor::zeros(vec![2, 4]);
        matmul(&a, &w);
    }

    #[test]
    fn rmsnorm_hand_computed() {
        // x = [3,4]: mean(x^2) = (9+16)/2 = 12.5, rms = sqrt(12.5) = 3.53553
        // y = [3/3.53553 * 2, 4/3.53553 * 0.5] = [1.69706, 0.56569]
        let x = Tensor::from_vec(vec![3., 4.], vec![1, 2]);
        let w = Tensor::from_vec(vec![2., 0.5], vec![2]);
        let y = rmsnorm(&x, &w, 0.0);
        assert_close(&y.data, &[1.69706, 0.56569], 1e-5);
    }

    #[test]
    fn rmsnorm_zero_row_stays_finite() {
        // eps inside the sqrt: denominator is sqrt(0 + eps), not 0
        let x = Tensor::from_vec(vec![0., 0., 0.], vec![1, 3]);
        let w = Tensor::from_vec(vec![1., 1., 1.], vec![3]);
        let y = rmsnorm(&x, &w, 1e-5);
        assert!(y.data.iter().all(|v| v.is_finite()));
        assert_eq!(y.data, vec![0., 0., 0.]);
    }

    #[test]
    fn softmax_hand_computed() {
        // row [0, ln 2]: exp = [1, 2], sum = 3 -> [1/3, 2/3]
        let mut x = Tensor::from_vec(vec![0., 2f32.ln()], vec![1, 2]);
        softmax(&mut x);
        assert_close(&x.data, &[1.0 / 3.0, 2.0 / 3.0], 1e-6);
    }

    #[test]
    fn softmax_survives_large_logits() {
        // exp(1001) overflows f32; max subtraction makes it exp(0) and exp(-1)
        let mut x = Tensor::from_vec(vec![1000., 1001.], vec![1, 2]);
        softmax(&mut x);
        assert!(x.data.iter().all(|v| v.is_finite()));
        let sum: f32 = x.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(x.data[1] > x.data[0]);
    }

    #[test]
    fn silu_and_swiglu_known_points() {
        // silu(0) = 0, silu(1) = 1/(1+e^-1) = 0.731059, silu(-1) = -0.268941
        let x = Tensor::from_vec(vec![0., 1., -1.], vec![3]);
        let y = silu(&x);
        assert_close(&y.data, &[0.0, 0.731059, -0.268941], 1e-5);

        // swiglu(gate=1, up=2) = silu(1) * 2 = 1.462117
        let gate = Tensor::from_vec(vec![1.], vec![1]);
        let up = Tensor::from_vec(vec![2.], vec![1]);
        assert_close(&swiglu(&gate, &up).data, &[1.462117], 1e-5);
    }

    #[test]
    fn add_mul_elementwise() {
        let a = Tensor::from_vec(vec![1., 2.], vec![2]);
        let b = Tensor::from_vec(vec![10., 0.5], vec![2]);
        assert_eq!(add(&a, &b).data, vec![11., 2.5]);
        assert_eq!(mul(&a, &b).data, vec![10., 1.]);
    }

    #[test]
    fn from_gguf_reverses_dims_to_row_major() {
        // uses the real model when present; skips otherwise so CI-like runs
        // without the 531MB file still pass
        let path = Path::new("../models/qwen2.5-0.5b-instruct-q8_0.gguf");
        if !path.exists() {
            eprintln!("skipped: model file not present");
            return;
        }
        let file = GgufFile::open(path).unwrap();
        let t = Tensor::from_gguf(&file, "blk.0.ffn_gate.weight").unwrap();
        // GGUF dims [896, 4864] (in, out in ggml order) -> shape [4864, 896]
        assert_eq!(t.shape, vec![4864, 896]);
        assert_eq!(t.strides, vec![896, 1]);
        assert_eq!(t.numel(), 4864 * 896);
    }
}
