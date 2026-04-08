//! Pre-computed GPU tensors for on-device quantization operations.
//!
//! [`GpuPrecomputed`] is created once per cache lifetime and reused for every
//! quantize/dequantize/attention step. All tensors live on the same device.

use candle_core::{Device, Result, Tensor};

use super::cache_err;
use super::config::{QuantNormMode, DEFAULT_ROTATION_SEED, QUANT_BLOCK_SIZE};

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
    pub fn new(
        bits: u8,
        head_dim: usize,
        norm_mode: QuantNormMode,
        outlier_blocks: usize,
        qjl_enabled: bool,
        device: &Device,
    ) -> Result<Self> {
        let block_dim = QUANT_BLOCK_SIZE;
        let polar_bits = bits - 1;

        let (rotation_fwd, rotation_inv) = build_rotation_matrices(block_dim, device)?;
        let (centroids, boundaries, outlier_centroids, outlier_boundaries) =
            build_codebooks(polar_bits, block_dim, norm_mode, device)?;

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

/// Build normalized Hadamard rotation matrices (forward + inverse).
fn build_rotation_matrices(block_dim: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let h_matrix = build_hadamard_matrix(block_dim, device)?;

    // Sign pattern: hardcoded llama.cpp pattern for block_size=32,
    // fallback to turboquant-rs for other sizes.
    let signs: Vec<f32> = if block_dim == QUANT_BLOCK_SIZE {
        vec![
            1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0,
            -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0,
            -1.0, -1.0, 1.0, -1.0,
        ]
    } else {
        crate::rotation::generate_sign_pattern(block_dim, DEFAULT_ROTATION_SEED)
    };
    let sign_tensor = Tensor::from_vec(signs, (1, block_dim), device)?;

    let rotation_fwd = h_matrix.broadcast_mul(&sign_tensor)?;
    let rotation_inv = rotation_fwd.t()?.contiguous()?;
    Ok((rotation_fwd, rotation_inv))
}

/// Build codebook tensors (normal + outlier) for the given normalization mode.
fn build_codebooks(
    polar_bits: u8,
    block_dim: usize,
    norm_mode: QuantNormMode,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    match norm_mode {
        QuantNormMode::L2Norm => build_codebooks_l2(polar_bits, block_dim, device),
        QuantNormMode::MaxNorm => build_codebooks_maxnorm(polar_bits, device),
    }
}

/// L2Norm codebooks: Beta-distribution optimal (Paper Algorithm 1).
fn build_codebooks_l2(
    polar_bits: u8,
    block_dim: usize,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let cb = crate::codebook::get_codebook(polar_bits, block_dim)
        .map_err(|e| cache_err(format!("codebook error: {e}")))?;
    let c: Vec<f32> = cb.centroids.iter().map(|&v| v as f32).collect();
    let b: Vec<f32> = cb.boundaries.iter().map(|&v| v as f32).collect();

    let outlier_bits = polar_bits + 1;
    let ocb = crate::codebook::get_codebook(outlier_bits, block_dim)
        .map_err(|e| cache_err(format!("outlier codebook error: {e}")))?;
    let oc: Vec<f32> = ocb.centroids.iter().map(|&v| v as f32).collect();
    let ob: Vec<f32> = ocb.boundaries.iter().map(|&v| v as f32).collect();

    let c_len = c.len();
    let b_len = b.len();
    let oc_len = oc.len();
    let ob_len = ob.len();
    Ok((
        Tensor::from_vec(c, c_len, device)?,
        Tensor::from_vec(b, b_len, device)?,
        Tensor::from_vec(oc, oc_len, device)?,
        Tensor::from_vec(ob, ob_len, device)?,
    ))
}

/// MaxNorm codebooks: empirical values from llama.cpp TQ3_0.
fn build_codebooks_maxnorm(polar_bits: u8, device: &Device) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    // 3-bit (8 centroids): from llama.cpp tq3_centroids[]
    let c3: Vec<f32> = vec![
        -2.1573, -1.3336, -0.7434, -0.2428, 0.2428, 0.7434, 1.3336, 2.1573,
    ];
    let b3: Vec<f32> = vec![-1.7455, -1.0385, -0.4931, 0.0, 0.4931, 1.0385, 1.7455];
    // 2-bit: every-other centroid from 3-bit
    let c2: Vec<f32> = vec![-2.1573, -0.7434, 0.7434, 2.1573];
    let b2: Vec<f32> = vec![-1.0385, 0.0, 1.0385];
    // 4-bit: interpolate within 3-bit range
    let c4: Vec<f32> = (0..16)
        .map(|i| -2.1573 + (i as f32) * (2.0 * 2.1573 / 15.0))
        .collect();
    let b4: Vec<f32> = (0..15)
        .map(|i| -2.1573 + (i as f32 + 0.5) * (2.0 * 2.1573 / 15.0))
        .collect();

    let (cn, bn) = match polar_bits {
        2 => (c2.clone(), b2.clone()),
        3 => (c3.clone(), b3.clone()),
        4 => (c4.clone(), b4.clone()),
        _ => (c3.clone(), b3.clone()),
    };
    let outlier_bits = polar_bits + 1;
    let (co, bo) = match outlier_bits {
        3 => (c3, b3),
        4 => (c4.clone(), b4.clone()),
        _ => (c4, b4),
    };
    let cn_len = cn.len();
    let bn_len = bn.len();
    let co_len = co.len();
    let bo_len = bo.len();
    Ok((
        Tensor::from_vec(cn, cn_len, device)?,
        Tensor::from_vec(bn, bn_len, device)?,
        Tensor::from_vec(co, co_len, device)?,
        Tensor::from_vec(bo, bo_len, device)?,
    ))
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
