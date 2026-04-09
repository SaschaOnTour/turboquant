//! L2Norm and MaxNorm codebook tensor construction.

use candle_core::{Device, Result, Tensor};

use crate::cache::cache_err;
use crate::cache::config::QuantNormMode;

/// MaxNorm codebook range limit (from llama.cpp tq3_centroids).
const MAXNORM_RANGE: f32 = 2.1573;
/// Number of centroids for 4-bit MaxNorm quantization.
const MAXNORM_4BIT_CENTROIDS: usize = 16;
/// Number of boundaries for 4-bit MaxNorm quantization.
const MAXNORM_4BIT_BOUNDARIES: usize = 15;
/// Midpoint offset for boundary interpolation.
const BOUNDARY_MIDPOINT: f32 = 0.5;

/// Build codebook tensors (normal + outlier) for the given normalization mode.
pub(super) fn build_codebooks(
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

    vecs_to_codebook_tensors(c, b, oc, ob, device)
}

/// MaxNorm codebooks: empirical values from llama.cpp TQ3_0.
fn build_codebooks_maxnorm(
    polar_bits: u8,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    // 3-bit (8 centroids): from llama.cpp tq3_centroids[]
    let c3: Vec<f32> = vec![
        -2.1573, -1.3336, -0.7434, -0.2428, 0.2428, 0.7434, 1.3336, 2.1573,
    ];
    let b3: Vec<f32> = vec![-1.7455, -1.0385, -0.4931, 0.0, 0.4931, 1.0385, 1.7455];
    // 2-bit: every-other centroid from 3-bit
    let c2: Vec<f32> = vec![-MAXNORM_RANGE, -0.7434, 0.7434, MAXNORM_RANGE];
    let b2: Vec<f32> = vec![-1.0385, 0.0, 1.0385];
    // 4-bit: interpolate within 3-bit range
    let full_range = 2.0 * MAXNORM_RANGE;
    let c4: Vec<f32> = (0..MAXNORM_4BIT_CENTROIDS)
        .map(|i| -MAXNORM_RANGE + (i as f32) * (full_range / MAXNORM_4BIT_BOUNDARIES as f32))
        .collect();
    let b4: Vec<f32> = (0..MAXNORM_4BIT_BOUNDARIES)
        .map(|i| {
            -MAXNORM_RANGE
                + (i as f32 + BOUNDARY_MIDPOINT) * (full_range / MAXNORM_4BIT_BOUNDARIES as f32)
        })
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
    vecs_to_codebook_tensors(cn, bn, co, bo, device)
}

/// Convert 4 codebook Vec<f32> into GPU Tensors.
fn vecs_to_codebook_tensors(
    c: Vec<f32>,
    b: Vec<f32>,
    oc: Vec<f32>,
    ob: Vec<f32>,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
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
