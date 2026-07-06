// Dequantization kernels: quantized on-disk formats -> f32.
//
// Q8_0 block (34 bytes, 32 elements):
//   bytes 0..2   f16 scale d, little-endian
//   bytes 2..34  32 x int8 quants
//   y[i] = f32(d) * q[i]

use crate::gguf::GgmlType;
use anyhow::{bail, ensure, Result};
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
    fn f32_passthrough_reads_le() {
        let mut raw = Vec::new();
        for v in [1.5f32, -2.25, 0.0] {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let out = dequantize(GgmlType::F32, &raw, 3).unwrap();
        assert_eq!(out, vec![1.5, -2.25, 0.0]);
    }
}
