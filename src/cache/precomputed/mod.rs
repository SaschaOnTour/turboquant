//! Pre-computed GPU tensors for on-device quantization operations.
//!
//! [`GpuPrecomputed`] is created once per cache lifetime and reused for every
//! quantize/dequantize/attention step. All tensors live on the same device.

mod codebooks;
mod rotation;

use candle_core::{Device, Result, Tensor};

use super::config::{CacheConfig, QUANT_BLOCK_SIZE};

/// Pre-computed tensors for on-device TurboQuant operations.
///
/// Rotation matrices are `[QUANT_BLOCK_SIZE x QUANT_BLOCK_SIZE]`, not
/// `[head_dim x head_dim]`. Each 32-element block within a head_dim=128
/// vector gets its own independent rotation + norm + quantization.
pub struct GpuPrecomputed {
    /// Forward rotation: H_normalized @ diag(signs), shape `[block_size, block_size]`.
    pub rotation_fwd: Tensor,
    /// Inverse rotation: `rotation_fwd^T`, shape `[block_size, block_size]`.
    pub rotation_inv: Tensor,
    /// Normal codebook centroids (polar_bits), shape `[n_centroids]`.
    pub centroids: Tensor,
    /// Normal codebook boundaries, shape `[n_boundaries]`.
    pub boundaries: Tensor,
    /// Outlier codebook centroids (polar_bits + 1), shape `[n_outlier_centroids]`.
    pub outlier_centroids: Tensor,
    /// Outlier codebook boundaries, shape `[n_outlier_boundaries]`.
    pub outlier_boundaries: Tensor,
    /// Max value of outlier centroids (cached to avoid per-call GPU->CPU sync).
    pub outlier_outer_centroid: f64,
    /// Pre-computed scale sign tensor for outlier block marking, shape `[1, num_blocks]`.
    pub scale_sign_tensor: Tensor,
    /// Pre-computed Rademacher matrix for QJL correction, shape `[dim, dim]`.
    /// `None` for PQ/PQO modes (no QJL).
    pub qjl_rademacher: Option<Tensor>,
}

impl GpuPrecomputed {
    /// Create pre-computed tensors for the given configuration.
    ///
    /// This is an integration function: it orchestrates Hadamard matrix
    /// construction, codebook setup, and sign pattern generation.
    pub fn new(config: &CacheConfig, device: &Device) -> Result<Self> {
        if config.bits < 3 || config.bits > 4 {
            return Err(super::cache_err(format!(
                "unsupported bits={}, expected 3 or 4",
                config.bits
            )));
        }
        const TQ_MAX_HEAD_DIM: usize = 1024;
        if config.head_dim > TQ_MAX_HEAD_DIM {
            return Err(super::cache_err(format!(
                "head_dim {} exceeds TQ_MAX_HEAD_DIM ({}). \
                 CUDA shared memory buffer overflow would occur.",
                config.head_dim, TQ_MAX_HEAD_DIM
            )));
        }
        let block_dim = QUANT_BLOCK_SIZE;
        let polar_bits = config.bits - 1;
        let head_dim = config.head_dim;
        let outlier_blocks = config.outlier_blocks;
        let norm_mode = config.norm_mode;
        let qjl_enabled = config.qjl_enabled();

        let (rotation_fwd, rotation_inv) = rotation::build_rotation_matrices(block_dim, device)?;
        let (centroids, boundaries, outlier_centroids, outlier_boundaries) =
            codebooks::build_codebooks(polar_bits, block_dim, norm_mode, device)?;

        let outlier_outer_centroid = outlier_centroids.max(0)?.to_scalar::<f32>()? as f64;
        let scale_sign_tensor = build_scale_sign_tensor(head_dim, outlier_blocks, device)?;

        let qjl_rademacher = if qjl_enabled {
            Some(build_rademacher_matrix(head_dim, device)?)
        } else {
            None
        };

        Ok(Self {
            rotation_fwd,
            rotation_inv,
            centroids,
            boundaries,
            outlier_centroids,
            outlier_boundaries,
            outlier_outer_centroid,
            scale_sign_tensor,
            qjl_rademacher,
        })
    }
}

/// Build scale-sign tensor for outlier block marking.
fn build_scale_sign_tensor(
    head_dim: usize,
    outlier_blocks: usize,
    device: &Device,
) -> Result<Tensor> {
    let num_blocks = head_dim / QUANT_BLOCK_SIZE;
    let effective_outlier = outlier_blocks.min(num_blocks);
    let mut signs = vec![1.0_f32; num_blocks];
    for sign in signs.iter_mut().take(effective_outlier) {
        *sign = -1.0;
    }
    Tensor::from_vec(signs, (1, num_blocks), device)
}

/// Build Rademacher projection matrix for QJL correction.
fn build_rademacher_matrix(head_dim: usize, device: &Device) -> Result<Tensor> {
    use super::config::DEFAULT_QJL_SEED;
    let mut rdata = Vec::with_capacity(head_dim * head_dim);
    for row in 0..head_dim {
        let row_vec = crate::qjl::generate_rademacher_row(head_dim, DEFAULT_QJL_SEED, row);
        rdata.extend_from_slice(&row_vec);
    }
    Tensor::from_vec(rdata, (head_dim, head_dim), device)
}
