//! Walsh-Hadamard Transform (WHT) butterfly on Candle tensors.
//!
//! CPU-only: on GPU, cuBLAS matmul is faster for the 32x32 block size.
//! The butterfly algorithm is O(N log N) vs O(N^2) for matmul, which
//! matters at larger block sizes but is negligible at block_size=32.

use candle_core::{DType, Device, Result, Tensor};

/// Inverse WHT rotation using butterfly algorithm.
///
/// Equivalent to `dequant.matmul(&rotation_inv)` where
/// `rotation_inv = diag(signs) @ H_normalized`.
///
/// For each row: `result = WHT(row * signs) / sqrt(block_size)`.
pub fn butterfly_wht_inverse_cpu(
    dequant: &Tensor,
    rotation_fwd: &Tensor,
    block_size: usize,
) -> Result<Tensor> {
    let (m, bs) = dequant.dims2()?;
    if bs != block_size {
        candle_core::bail!(
            "butterfly_wht_inverse_cpu: block_size mismatch (tensor={bs}, expected={block_size})"
        );
    }

    // Extract sign pattern: rotation_fwd[0][j] = signs[j] / sqrt(N)
    let sqrt_n = (block_size as f32).sqrt();
    let signs: Vec<f32> = rotation_fwd
        .narrow(0, 0, 1)?
        .squeeze(0)?
        .to_dtype(DType::F32)?
        .to_vec1()?;

    let mut data: Vec<f32> = dequant.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let inv_sqrt_n = 1.0 / sqrt_n;

    for block in data.chunks_mut(block_size) {
        // Step 1: Apply sign flip (undo signs from forward rotation)
        for j in 0..block_size {
            block[j] *= signs[j] * sqrt_n;
        }

        // Step 2: Butterfly WHT (unnormalized, 5 stages for block_size=32)
        let mut h = 1;
        while h < block_size {
            let full = h << 1;
            let mut i = 0;
            while i < block_size {
                for j in 0..h {
                    let a = block[i + j];
                    let b = block[i + j + h];
                    block[i + j] = a + b;
                    block[i + j + h] = a - b;
                }
                i += full;
            }
            h <<= 1;
        }

        // Step 3: Normalize by 1/sqrt(N)
        for val in block.iter_mut() {
            *val *= inv_sqrt_n;
        }
    }

    Tensor::from_vec(data, (m, block_size), &Device::Cpu)
}
