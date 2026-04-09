//! Hadamard and rotation matrix construction.

use candle_core::{Device, Result, Tensor};

use crate::cache::config::{DEFAULT_ROTATION_SEED, QUANT_BLOCK_SIZE};

/// Build normalized Hadamard rotation matrices (forward + inverse).
pub(super) fn build_rotation_matrices(
    block_dim: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let h_matrix = build_hadamard_matrix(block_dim, device)?;

    // Sign pattern: hardcoded llama.cpp pattern for block_size=32,
    // fallback to turboquant-rs for other sizes.
    let signs: Vec<f32> = if block_dim == QUANT_BLOCK_SIZE {
        vec![
            1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0,
            1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0,
        ]
    } else {
        crate::rotation::generate_sign_pattern(block_dim, DEFAULT_ROTATION_SEED)
    };
    let sign_tensor = Tensor::from_vec(signs, (1, block_dim), device)?;

    let rotation_fwd = h_matrix.broadcast_mul(&sign_tensor)?;
    let rotation_inv = rotation_fwd.t()?.contiguous()?;
    Ok((rotation_fwd, rotation_inv))
}

/// Build normalized Hadamard matrix of size `dim x dim`.
///
/// H_n = H / sqrt(dim) is orthogonal and self-inverse: H_n @ H_n = I.
fn build_hadamard_matrix(dim: usize, device: &Device) -> Result<Tensor> {
    let mut h = vec![1.0f32; 1];
    let mut size = 1;
    while size < dim {
        let new_size = size * 2;
        let mut new_h = vec![0.0f32; new_size * new_size];
        for i in 0..size {
            for j in 0..size {
                let val = h[i * size + j];
                new_h[i * new_size + j] = val;
                new_h[i * new_size + (j + size)] = val;
                new_h[(i + size) * new_size + j] = val;
                new_h[(i + size) * new_size + (j + size)] = -val;
            }
        }
        h = new_h;
        size = new_size;
    }
    let norm = 1.0 / (dim as f32).sqrt();
    for v in h.iter_mut() {
        *v *= norm;
    }
    Tensor::from_vec(h, (dim, dim), device)
}
