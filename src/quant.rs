// Dequantization kernels: quantized on-disk formats -> f32.
//
// Q8_0 block (34 bytes, 32 elements):
//   bytes 0..2   f16 scale d, little-endian
//   bytes 2..34  32 x int8 quants
//   y[i] = f32(d) * q[i]

use crate::gguf::GgmlType;
use anyhow::{bail, ensure, Context, Result};
use half::f16;

/// Elements per Q8_0 block.
pub const QK8_0: usize = 32;
/// Bytes per Q8_0 block: 2 (f16 scale) + 32 (int8 quants).
pub const Q8_0_BLOCK_BYTES: usize = 2 + QK8_0;

pub fn dequantize_q8_0(raw: &[u8], n_elements: usize) -> Result<Vec<f32>> {
    ensure!(
        n_elements % QK8_0 == 0,
        "Q8_0 element count {} is not a multiple of {}",
        n_elements,
        QK8_0
    );
    let n_blocks = n_elements / QK8_0;
    ensure!(
        raw.len() == n_blocks * Q8_0_BLOCK_BYTES,
        "Q8_0 data is {} bytes, expected {} ({} blocks of {})",
        raw.len(),
        n_blocks * Q8_0_BLOCK_BYTES,
        n_blocks,
        Q8_0_BLOCK_BYTES
    );

    let mut out = Vec::with_capacity(n_elements);
    for block in raw.chunks_exact(Q8_0_BLOCK_BYTES) {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        for &q in &block[2..] {
            out.push(d * (q as i8) as f32);
        }
    }
    Ok(out)
}

/// Elements per Q5_0 block.
pub const QK5_0: usize = 32;
/// Q5_0 block: f16 scale + 4 bytes of packed high bits + 16 nibble bytes.
pub const Q5_0_BLOCK_BYTES: usize = 2 + 4 + 16;

/// Dequantize one Q5_0 block into out[..32]. The fifth quant bit for
/// element j lives at bit j of the 32-bit qh field; elements 0..16 take the
/// low nibbles of the 16 quant bytes, 16..32 the high nibbles.
/// y = d * (5-bit quant - 16).
fn dequant_block_q5_0(block: &[u8], out: &mut [f32]) {
    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
    let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
    let qs = &block[6..22];
    for j in 0..16 {
        let hi0 = ((qh >> j) & 1) << 4;
        let hi1 = ((qh >> (j + 16)) & 1) << 4;
        out[j] = d * (((qs[j] & 0x0F) as u32 | hi0) as i32 - 16) as f32;
        out[j + 16] = d * (((qs[j] >> 4) as u32 | hi1) as i32 - 16) as f32;
    }
}

/// Elements per K-quant superblock.
pub const QK_K: usize = 256;
/// Q4_K superblock: f16 d + f16 dmin + 12 packed 6-bit scale/min bytes
/// + 128 bytes of 4-bit quants.
pub const Q4_K_BLOCK_BYTES: usize = 2 + 2 + 12 + 128;
/// Q6_K superblock: 128 bytes low 4 bits + 64 bytes high 2 bits
/// + 16 signed sub-scales + f16 d.
pub const Q6_K_BLOCK_BYTES: usize = 128 + 64 + 16 + 2;

/// Unpack sub-block j's 6-bit (scale, min) pair from the 12 packed bytes.
/// Entries 0-3 sit in the low 6 bits of bytes 0-3 (scales) and 4-7 (mins);
/// entries 4-7 are split: low 4 bits in bytes 8-11, high 2 bits in the
/// leftover top bits of bytes 0-7.
fn q4k_scale_min(j: usize, scales: &[u8]) -> (f32, f32) {
    let (sc, m) = if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    };
    (sc as f32, m as f32)
}

/// Dequantize one Q4_K superblock into out[..256].
/// y = (d * sc_j) * q - (dmin * m_j) per 32-element sub-block j; each group
/// of 64 elements shares 32 quant bytes (low nibbles then high nibbles).
fn dequant_block_q4_k(block: &[u8], out: &mut [f32]) {
    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
    let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
    let scales = &block[4..16];
    let qs = &block[16..144];
    for j64 in 0..QK_K / 64 {
        let q = &qs[32 * j64..32 * (j64 + 1)];
        let (sc1, m1) = q4k_scale_min(2 * j64, scales);
        let (sc2, m2) = q4k_scale_min(2 * j64 + 1, scales);
        let (d1, min1) = (d * sc1, dmin * m1);
        let (d2, min2) = (d * sc2, dmin * m2);
        for l in 0..32 {
            out[64 * j64 + l] = d1 * (q[l] & 0x0F) as f32 - min1;
            out[64 * j64 + 32 + l] = d2 * (q[l] >> 4) as f32 - min2;
        }
    }
}

/// Dequantize one Q6_K superblock into out[..256].
/// q = 6-bit value from (4 low bits | 2 high bits) - 32, y = d * sc * q with
/// 16 sub-blocks of 16 elements and signed 8-bit sub-scales.
fn dequant_block_q6_k(block: &[u8], out: &mut [f32]) {
    let d = f16::from_le_bytes([block[208], block[209]]).to_f32();
    for half in 0..2 {
        let ql = &block[64 * half..];
        let qh = &block[128 + 32 * half..];
        let sc = &block[192 + 8 * half..];
        let y = &mut out[128 * half..];
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 0x0F) as i32 | (((qh[l] as i32) & 3) << 4)) - 32;
            let q2 = ((ql[l + 32] & 0x0F) as i32 | ((((qh[l] as i32) >> 2) & 3) << 4)) - 32;
            let q3 = ((ql[l] >> 4) as i32 | ((((qh[l] as i32) >> 4) & 3) << 4)) - 32;
            let q4 = ((ql[l + 32] >> 4) as i32 | ((((qh[l] as i32) >> 6) & 3) << 4)) - 32;
            y[l] = d * (sc[is] as i8) as f32 * q1 as f32;
            y[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
            y[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
            y[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
        }
    }
}

/// (elements per block, bytes per block, block dequantizer) for dtypes
/// decoded via the generic block path. Q8_0 has its own specialized kernel.
pub fn block_dequant(dtype: GgmlType) -> Option<(usize, usize, fn(&[u8], &mut [f32]))> {
    match dtype {
        GgmlType::Q5_0 => Some((QK5_0, Q5_0_BLOCK_BYTES, dequant_block_q5_0)),
        GgmlType::Q4K => Some((QK_K, Q4_K_BLOCK_BYTES, dequant_block_q4_k)),
        GgmlType::Q6K => Some((QK_K, Q6_K_BLOCK_BYTES, dequant_block_q6_k)),
        _ => None,
    }
}

/// Dequantize a whole tensor of any block-dequantizable dtype.
fn dequantize_blocks(raw: &[u8], n_elements: usize, dtype: GgmlType) -> Result<Vec<f32>> {
    let (block_elems, block_bytes, f) =
        block_dequant(dtype).with_context(|| format!("no block dequantizer for {:?}", dtype))?;
    ensure!(
        n_elements % block_elems == 0,
        "element count {} is not a multiple of {}",
        n_elements,
        block_elems
    );
    let n_blocks = n_elements / block_elems;
    ensure!(
        raw.len() == n_blocks * block_bytes,
        "data is {} bytes, expected {} ({} blocks of {})",
        raw.len(), n_blocks * block_bytes, n_blocks, block_bytes
    );
    let mut out = vec![0.0f32; n_elements];
    for (block, chunk) in raw.chunks_exact(block_bytes).zip(out.chunks_exact_mut(block_elems)) {
        f(block, chunk);
    }
    Ok(out)
}

/// Fused block-quant matmul for any dtype with a block dequantizer.
pub fn matmul_quant(
    a: &crate::tensor::Tensor,
    raw: &[u8],
    d_out: usize,
    d_in: usize,
    dtype: GgmlType,
) -> crate::tensor::Tensor {
    // one monomorphized instantiation per dtype so the block dequantizer
    // inlines into the kernel loop (a fn pointer would be an indirect call
    // per block)
    match dtype {
        GgmlType::Q8_0 => matmul_q8_0(a, raw, d_out, d_in),
        GgmlType::Q5_0 => {
            matmul_blocks(a, raw, d_out, d_in, QK5_0, Q5_0_BLOCK_BYTES, dequant_block_q5_0)
        }
        GgmlType::Q4K => {
            matmul_blocks(a, raw, d_out, d_in, QK_K, Q4_K_BLOCK_BYTES, dequant_block_q4_k)
        }
        GgmlType::Q6K => {
            matmul_blocks(a, raw, d_out, d_in, QK_K, Q6_K_BLOCK_BYTES, dequant_block_q6_k)
        }
        other => panic!("no fused matmul kernel for {:?}", other),
    }
}

/// Shared fused kernel: dequantize one block at a time into a stack buffer
/// (L1-resident, never in RAM) and dot it against the activation slice.
/// RAM traffic stays at quantized size.
fn matmul_blocks<F>(
    a: &crate::tensor::Tensor,
    raw: &[u8],
    d_out: usize,
    d_in: usize,
    block_elems: usize,
    block_bytes: usize,
    dequant: F,
) -> crate::tensor::Tensor
where
    F: Fn(&[u8], &mut [f32]) + Sync,
{
    assert!(d_in % block_elems == 0, "matmul: in dim {} not a multiple of {}", d_in, block_elems);
    let row_bytes = d_in / block_elems * block_bytes;
    assert_eq!(raw.len(), d_out * row_bytes, "matmul: bad weight byte count");

    crate::tensor::matmul_with(a, d_out, d_in, |o, a_row| {
        let row = &raw[o * row_bytes..(o + 1) * row_bytes];
        let mut buf = [0.0f32; QK_K]; // large enough for every block size
        let mut acc = 0.0f32;
        for (b, block) in row.chunks_exact(block_bytes).enumerate() {
            dequant(block, &mut buf[..block_elems]);
            acc += crate::tensor::dot(
                &a_row[b * block_elems..(b + 1) * block_elems],
                &buf[..block_elems],
            );
        }
        acc
    })
}

/// Fused Q8_0 matmul: activations [seq, in] x quantized weight rows
/// [out, in] -> [seq, out]. Dequantization happens inside the dot product;
/// no f32 weight row is ever materialized. Accumulation is f32 within a
/// 32-element block (8 independent lanes), then f32 across blocks after
/// scaling -- the same products as dequantize-then-dot, grouped by block.
pub fn matmul_q8_0(a: &crate::tensor::Tensor, raw: &[u8], d_out: usize, d_in: usize) -> crate::tensor::Tensor {
    assert_eq!(d_in % QK8_0, 0, "Q8_0 matmul: in dim {} not a multiple of {}", d_in, QK8_0);
    let row_bytes = d_in / QK8_0 * Q8_0_BLOCK_BYTES;
    assert_eq!(raw.len(), d_out * row_bytes, "Q8_0 matmul: bad weight byte count");

    crate::tensor::matmul_with(a, d_out, d_in, |o, a_row| {
        let row = &raw[o * row_bytes..(o + 1) * row_bytes];
        let mut acc = 0.0f32;
        for (b, block) in row.chunks_exact(Q8_0_BLOCK_BYTES).enumerate() {
            let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
            // fixed-size array refs give LLVM compile-time bounds so the
            // whole 32-element block body unrolls and vectorizes
            let qs: &[u8; QK8_0] = block[2..].try_into().unwrap();
            let a_block: &[f32; QK8_0] = a_row[b * QK8_0..(b + 1) * QK8_0].try_into().unwrap();
            let mut lanes = [0.0f32; 8];
            for j in 0..QK8_0 / 8 {
                for l in 0..8 {
                    lanes[l] += a_block[j * 8 + l] * (qs[j * 8 + l] as i8) as f32;
                }
            }
            acc += scale * lanes.iter().sum::<f32>();
        }
        acc
    })
}

/// Decode any supported tensor dtype to f32.
pub fn dequantize(dtype: GgmlType, raw: &[u8], n_elements: usize) -> Result<Vec<f32>> {
    match dtype {
        GgmlType::F32 => {
            ensure!(
                raw.len() == n_elements * 4,
                "F32 data is {} bytes, expected {}",
                raw.len(),
                n_elements * 4
            );
            Ok(raw
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect())
        }
        GgmlType::Q8_0 => dequantize_q8_0(raw, n_elements),
        GgmlType::Q5_0 | GgmlType::Q4K | GgmlType::Q6K => {
            dequantize_blocks(raw, n_elements, dtype)
        }
        other => bail!("dequantization for {:?} is not implemented", other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serialize one Q8_0 block from a scale and 32 quants.
    fn block(scale: f32, quants: [i8; 32]) -> Vec<u8> {
        let mut b = f16::from_f32(scale).to_le_bytes().to_vec();
        b.extend(quants.iter().map(|&q| q as u8));
        b
    }

    #[test]
    fn one_block_with_negative_quants() {
        // scale 0.5 is exactly representable in f16, quants -16..16,
        // so every output is exact: y[i] = 0.5 * (i - 16)
        let quants: [i8; 32] = std::array::from_fn(|i| i as i8 - 16);
        let raw = block(0.5, quants);
        let out = dequantize_q8_0(&raw, 32).unwrap();
        for (i, &y) in out.iter().enumerate() {
            assert_eq!(y, 0.5 * (i as f32 - 16.0), "element {}", i);
        }
    }

    #[test]
    fn zero_scale_gives_exact_zeros() {
        let raw = block(0.0, [127; 32]);
        let out = dequantize_q8_0(&raw, 32).unwrap();
        assert!(out.iter().all(|&y| y == 0.0));
    }

    #[test]
    fn two_blocks_use_their_own_scales() {
        let mut raw = block(1.0, [3; 32]);
        raw.extend(block(2.0, [3; 32]));
        let out = dequantize_q8_0(&raw, 64).unwrap();
        assert_eq!(out[0], 3.0);
        assert_eq!(out[32], 6.0);
    }

    #[test]
    fn rejects_wrong_length() {
        let raw = block(1.0, [0; 32]);
        assert!(dequantize_q8_0(&raw[..33], 32).is_err()); // truncated block
        assert!(dequantize_q8_0(&raw, 64).is_err()); // count/bytes mismatch
        assert!(dequantize_q8_0(&raw, 31).is_err()); // not a multiple of 32
    }

    #[test]
    fn fused_q8_matmul_matches_dequant_reference() {
        // weight: 3 output rows x 64 inputs = 2 blocks per row, varied
        // scales and signed quants; the fused kernel must agree with
        // dequantize-then-f32-matmul to f32 rounding
        let mut raw = Vec::new();
        for o in 0..3i32 {
            for b in 0..2i32 {
                let quants: [i8; 32] =
                    std::array::from_fn(|j| ((j as i32 * 7 + o * 13 + b * 5) % 255 - 127) as i8);
                raw.extend(block(0.5 + o as f32 * 0.25, quants));
            }
        }
        let x = crate::tensor::Tensor::from_vec(
            (0..2 * 64).map(|i| ((i as f32) * 0.31).sin()).collect(),
            vec![2, 64],
        );

        let weights_f32 = dequantize_q8_0(&raw, 3 * 64).unwrap();
        let w = crate::tensor::Tensor::from_vec(weights_f32, vec![3, 64]);
        let reference = crate::tensor::matmul(&x, &w);
        let fused = matmul_q8_0(&x, &raw, 3, 64);

        assert_eq!(fused.shape, reference.shape);
        for (f, r) in fused.data.iter().zip(&reference.data) {
            assert!((f - r).abs() < 1e-4, "fused {} vs reference {}", f, r);
        }
    }

    #[test]
    fn q4_k_block_hand_computed() {
        // d = 1.0 (f16 0x3C00), dmin = 0.5 (f16 0x3800)
        // sub-block 0: sc=3 (scales[0]), m=2 (scales[4])
        // sub-block 1: sc=5 (scales[1]), m=1 (scales[5])
        // sub-block 4: split encoding, low 4 bits in scales[8]=7, high bits 0
        // qs[0] = 0x51: elem 0 gets low nibble 1, elem 32 gets high nibble 5
        // qs[64] = 0x04: elem 128 (sub-block 4) gets low nibble 4
        let mut block = vec![0u8; Q4_K_BLOCK_BYTES];
        block[0..2].copy_from_slice(&[0x00, 0x3C]);
        block[2..4].copy_from_slice(&[0x00, 0x38]);
        block[4] = 3; // scales[0]: sc for sub-block 0
        block[8] = 2; // scales[4]: m for sub-block 0
        block[5] = 5; // scales[1]: sc for sub-block 1
        block[9] = 1; // scales[5]: m for sub-block 1
        block[12] = 7; // scales[8]: low 4 bits of sc for sub-block 4
        block[16] = 0x51; // qs[0]
        block[16 + 64] = 0x04; // qs[64]
        let mut out = [0.0f32; QK_K];
        dequant_block_q4_k(&block, &mut out);
        // y[0]   = 1.0*3*1 - 0.5*2 = 2.0
        assert_eq!(out[0], 2.0);
        // y[32]  = 1.0*5*5 - 0.5*1 = 24.5
        assert_eq!(out[32], 24.5);
        // y[128] = 1.0*7*4 - 0.5*0 = 28.0
        assert_eq!(out[128], 28.0);
        // an untouched element in sub-block 0: q=0 -> y = 0 - dmin*m0 = -1.0
        assert_eq!(out[1], -1.0);
    }

    #[test]
    fn q6_k_block_hand_computed() {
        // d = 0.5 (f16 0x3800) at bytes 208..210; signed sub-scales at
        // 192..208: sc[0]=2, sc[2]=1, sc[4]=3, sc[6]=-1
        // ql[0] = 0x21 (low nibble 1 -> q1, high nibble 2 -> q3), ql[32]=0
        // qh[0] = 0xC4: q1 high bits 0, q2 high bits 1, q3 0, q4 3
        let mut block = vec![0u8; Q6_K_BLOCK_BYTES];
        block[208..210].copy_from_slice(&[0x00, 0x38]);
        block[192] = 2;
        block[194] = 1;
        block[196] = 3;
        block[198] = (-1i8) as u8;
        block[0] = 0x21;
        block[128] = 0xC4;
        let mut out = [0.0f32; QK_K];
        dequant_block_q6_k(&block, &mut out);
        // q1 = (1 | 0<<4) - 32 = -31   y[0]  = 0.5 * 2 * -31 = -31
        assert_eq!(out[0], -31.0);
        // q2 = (0 | 1<<4) - 32 = -16   y[32] = 0.5 * 1 * -16 = -8
        assert_eq!(out[32], -8.0);
        // q3 = (2 | 0<<4) - 32 = -30   y[64] = 0.5 * 3 * -30 = -45
        assert_eq!(out[64], -45.0);
        // q4 = (0 | 3<<4) - 32 = 16    y[96] = 0.5 * -1 * 16 = -8
        assert_eq!(out[96], -8.0);
    }

    #[test]
    fn q5_0_block_hand_computed() {
        // d = 0.5; qs[0] = 0x51: elem 0 low nibble 1, elem 16 high nibble 5;
        // qh bit 0 set -> elem 0 gets its fifth bit, elem 16 does not
        let mut block = vec![0u8; Q5_0_BLOCK_BYTES];
        block[0..2].copy_from_slice(&[0x00, 0x38]);
        block[2] = 0b0000_0001; // qh low byte
        block[6] = 0x51; // first quant byte
        let mut out = [0.0f32; QK5_0];
        dequant_block_q5_0(&block, &mut out);
        // elem 0:  q = (1 | 16) - 16 = 1   -> 0.5
        assert_eq!(out[0], 0.5);
        // elem 16: q = 5 - 16 = -11        -> -5.5
        assert_eq!(out[16], -5.5);
        // elem 1:  q = 0 - 16 = -16        -> -8.0
        assert_eq!(out[1], -8.0);
    }

    #[test]
    fn fused_k_matmul_matches_dequant_reference() {
        // two Q4_K rows of one superblock each, pseudo-random bytes; the
        // fused kernel must agree with dequantize-then-matmul
        let raw: Vec<u8> = (0..2 * Q4_K_BLOCK_BYTES)
            .map(|i| ((i * 37 + 11) % 251) as u8)
            .collect();
        let x = crate::tensor::Tensor::from_vec(
            (0..3 * QK_K).map(|i| ((i as f32) * 0.17).sin()).collect(),
            vec![3, QK_K],
        );
        let w = crate::tensor::Tensor::from_vec(
            dequantize(GgmlType::Q4K, &raw, 2 * QK_K).unwrap(),
            vec![2, QK_K],
        );
        let reference = crate::tensor::matmul(&x, &w);
        let fused = matmul_quant(&x, &raw, 2, QK_K, GgmlType::Q4K);
        assert_eq!(fused.shape, reference.shape);
        for (f, r) in fused.data.iter().zip(&reference.data) {
            assert!((f - r).abs() <= 1e-3 * r.abs().max(1.0), "fused {} vs ref {}", f, r);
        }
    }

    #[test]
    fn f32_passthrough_reads_le() {
        let mut raw = Vec::new();
        for v in [1.5f32, -2.25, 0.0] {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let out = dequantize(GgmlType::F32, &raw, 3).unwrap();
        assert_eq!(out, vec![1.5, -2.25, 0.0]);
    }
}
