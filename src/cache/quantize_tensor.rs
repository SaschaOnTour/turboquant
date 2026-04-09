//! Block-level PolarQuant on Candle tensors.
//!
//! Operates on `[N, head_dim]` tensors, splitting each vector into
//! `QUANT_BLOCK_SIZE`-element blocks for independent quantization.
//!
//! CPU path uses turboquant-rs core functions (pure Rust).
//! GPU path (TODO: Phase 4c) will use CUDA kernels.

use candle_core::{DType, Device, Result, Tensor};

use super::cache_err;
use super::config::{QuantNormMode, BITS_PER_BYTE, QUANT_BLOCK_SIZE};
use super::precomputed::GpuPrecomputed;
use super::wht_tensor::butterfly_wht_inverse_cpu;

/// Shared quantization configuration for PolarQuant operations.
pub(crate) struct QuantConfig<'a> {
    pub head_dim: usize,
    pub bits: u8,
    pub outlier_blocks: usize,
    pub pre: &'a GpuPrecomputed,
}

impl QuantConfig<'_> {
    /// Number of quantization blocks per head_dim vector.
    pub fn num_blocks(&self) -> usize {
        self.head_dim / QUANT_BLOCK_SIZE
    }

    /// Packed dimension: bytes per token for indices.
    pub fn packed_dim(&self) -> usize {
        self.head_dim * self.bits as usize / BITS_PER_BYTE
    }
}

/// Quantize `[N, head_dim]` f32 input to packed indices + scales.
///
/// Returns `(indices: [N, packed_dim] U8, scales: [N, num_blocks] F16)`
/// where `packed_dim = head_dim * bits / 8` and `num_blocks = head_dim / QUANT_BLOCK_SIZE`.
pub fn polar_quantize(
    input: &Tensor,
    norm_mode: QuantNormMode,
    config: &QuantConfig<'_>,
) -> Result<(Tensor, Tensor)> {
    let n = input.dims()[0];
    let head_dim = config.head_dim;
    let bits = config.bits;
    let outlier_blocks = config.outlier_blocks;
    let pre = config.pre;
    let num_blocks = config.num_blocks();
    let packed_dim = config.packed_dim();

    // CUDA fast path: matmul WHT + fused quant-and-pack kernel
    #[cfg(feature = "cuda")]
    if input.device().is_cuda()
        && norm_mode == QuantNormMode::MaxNorm
        && outlier_blocks >= num_blocks
    {
        return super::cuda::quantize::cuda_quantize_fast(input, n, config);
    }

    let (rotated, safe_norm) = normalize_and_rotate(input, n, num_blocks, norm_mode, pre)?;

    let indices = bucketize_blocks(&rotated, n, num_blocks, outlier_blocks, pre)?;
    let indices = indices.reshape((n, head_dim))?;
    let scales = safe_norm.reshape((n, num_blocks))?;

    // Mark outlier blocks with negative scale
    let scales = scales.broadcast_mul(&pre.scale_sign_tensor)?;
    let scales = scales.to_dtype(DType::F16)?;

    // Pack indices from U8 to bit-packed
    let packed_indices = pack_indices_cpu(&indices, n, packed_dim, bits)?;

    Ok((packed_indices, scales))
}

/// Normalize input blocks and apply WHT rotation.
///
/// L2Norm: normalize → rotate. MaxNorm: rotate → scale by max-abs.
/// Returns `(rotated, norms)` both `[N * num_blocks, QUANT_BLOCK_SIZE]` / `[N * num_blocks, 1]`.
fn normalize_and_rotate(
    input: &Tensor,
    n: usize,
    num_blocks: usize,
    norm_mode: QuantNormMode,
    pre: &GpuPrecomputed,
) -> Result<(Tensor, Tensor)> {
    const MIN_NORM: f64 = 1e-10;
    let blocked = input.reshape((n * num_blocks, QUANT_BLOCK_SIZE))?;

    match norm_mode {
        QuantNormMode::L2Norm => {
            let norm = blocked
                .sqr()?
                .sum_keepdim(1)?
                .sqrt()?
                .clamp(MIN_NORM, f64::MAX)?;
            let normalized = blocked.broadcast_div(&norm)?;
            let rotated = normalized.matmul(&pre.rotation_fwd)?;
            Ok((rotated, norm))
        }
        QuantNormMode::MaxNorm => {
            let rotated_raw = blocked.matmul(&pre.rotation_fwd)?;
            let outer_c = pre.outlier_outer_centroid;
            let amax = rotated_raw.abs()?.max_keepdim(1)?;
            let scale = (amax / outer_c)?.clamp(MIN_NORM, f64::MAX)?;
            let rotated = rotated_raw.broadcast_div(&scale)?;
            Ok((rotated, scale))
        }
    }
}

/// Bucketize rotated blocks into codebook indices.
///
/// Splits blocks into outlier (higher-bit codebook) and normal blocks,
/// bucketizes each against the appropriate boundaries, then concatenates.
fn bucketize_blocks(
    rotated: &Tensor,
    n: usize,
    num_blocks: usize,
    outlier_blocks: usize,
    pre: &GpuPrecomputed,
) -> Result<Tensor> {
    let effective_outlier = outlier_blocks.min(num_blocks);
    let outlier_rows = n * effective_outlier;
    let normal_start = outlier_rows;
    let normal_rows = n * num_blocks - outlier_rows;

    let idx_out = if outlier_rows > 0 {
        Some(bucketize_slice(
            rotated,
            0,
            outlier_rows,
            &pre.outlier_boundaries,
        )?)
    } else {
        None
    };

    let idx_norm = if normal_rows > 0 {
        Some(bucketize_slice(
            rotated,
            normal_start,
            normal_rows,
            &pre.boundaries,
        )?)
    } else {
        None
    };

    match (idx_out, idx_norm) {
        (Some(o), Some(n_idx)) => Tensor::cat(&[&o, &n_idx], 0),
        (Some(o), None) => Ok(o),
        (None, Some(n_idx)) => Ok(n_idx),
        (None, None) => Err(cache_err("no blocks to quantize")),
    }
}

/// Bucketize a contiguous slice of rotated blocks against boundary values.
fn bucketize_slice(
    rotated: &Tensor,
    offset: usize,
    rows: usize,
    boundaries: &Tensor,
) -> Result<Tensor> {
    let n_b = boundaries.elem_count();
    let b_exp = boundaries.reshape((1, 1, n_b))?;
    let slice = rotated.narrow(0, offset, rows)?;
    slice
        .unsqueeze(2)?
        .broadcast_gt(&b_exp)?
        .to_dtype(DType::U8)?
        .sum_keepdim(2)?
        .squeeze(2)
}

/// Dequantize packed indices + scales back to `[N, head_dim]` f32.
pub fn polar_dequantize(
    indices: &Tensor,
    scales: &Tensor,
    config: &QuantConfig<'_>,
) -> Result<Tensor> {
    let n = indices.dims()[0];
    let head_dim = config.head_dim;
    let bits = config.bits;
    let outlier_blocks = config.outlier_blocks;
    let pre = config.pre;
    let num_blocks = config.num_blocks();

    // CUDA fast path: fused unpack + codebook + WHT + scale kernel
    #[cfg(feature = "cuda")]
    if indices.device().is_cuda() && outlier_blocks >= num_blocks {
        return super::cuda::quantize::cuda_dequantize_fast(indices, scales, n, config);
    }

    // CPU fallback: unpack + tensor ops
    let indices_unpacked = unpack_indices_on_device(indices, n, head_dim, bits)?;

    // Reshape to blocks
    let indices_blocked = indices_unpacked.reshape((n * num_blocks, QUANT_BLOCK_SIZE))?;
    let scales_blocked = scales.to_dtype(DType::F32)?.reshape((n * num_blocks, 1))?;
    let abs_scales = scales_blocked.abs()?;
    let indices_flat = indices_blocked.flatten_all()?.to_dtype(DType::U32)?;

    // Codebook lookup (select based on outlier config)
    let dequant = codebook_lookup(&indices_flat, &scales_blocked, n, config)?;

    // Inverse WHT rotation
    let reconstructed = if indices.device().is_cpu() {
        butterfly_wht_inverse_cpu(&dequant, &pre.rotation_fwd, QUANT_BLOCK_SIZE)?
    } else {
        dequant.matmul(&pre.rotation_inv)?
    };

    // Re-scale and reshape
    reconstructed
        .broadcast_mul(&abs_scales)?
        .reshape((n, head_dim))
}

/// Codebook lookup with outlier/normal block dispatch.
fn codebook_lookup(
    indices_flat: &Tensor,
    scales_blocked: &Tensor,
    n: usize,
    config: &QuantConfig<'_>,
) -> Result<Tensor> {
    let outlier_blocks = config.outlier_blocks;
    let num_blocks = config.num_blocks();
    let pre = config.pre;
    if outlier_blocks >= num_blocks {
        // All blocks use outlier codebook
        pre.outlier_centroids
            .index_select(indices_flat, 0)?
            .reshape((n * num_blocks, QUANT_BLOCK_SIZE))
    } else if outlier_blocks == 0 {
        // All blocks use normal codebook
        pre.centroids
            .index_select(indices_flat, 0)?
            .reshape((n * num_blocks, QUANT_BLOCK_SIZE))
    } else {
        // Mixed: negative scale = outlier codebook
        let is_outlier = scales_blocked
            .lt(0.0)?
            .to_dtype(DType::F32)?
            .broadcast_as((n * num_blocks, QUANT_BLOCK_SIZE))?;
        let n_nc = pre.centroids.elem_count() as u32;
        let clamped = indices_flat.clamp(0u32, n_nc - 1)?;
        let normal = pre
            .centroids
            .index_select(&clamped, 0)?
            .reshape((n * num_blocks, QUANT_BLOCK_SIZE))?;
        let outlier = pre
            .outlier_centroids
            .index_select(indices_flat, 0)?
            .reshape((n * num_blocks, QUANT_BLOCK_SIZE))?;
        let not_outlier = (1.0 - &is_outlier)?;
        (&is_outlier * &outlier)? + (&not_outlier * &normal)?
    }
}

/// Pack U8 indices to bit-packed format on CPU.
fn pack_indices_cpu(indices: &Tensor, n: usize, packed_dim: usize, bits: u8) -> Result<Tensor> {
    let indices_cpu: Vec<u8> = indices.to_device(&Device::Cpu)?.flatten_all()?.to_vec1()?;
    let packed = match bits {
        2 => crate::packed::pack_indices_2bit(&indices_cpu),
        3 => crate::packed::pack_indices_3bit(&indices_cpu),
        4 => crate::packed::pack_indices_4bit(&indices_cpu),
        _ => return Err(cache_err(format!("unsupported bits: {bits}"))),
    };
    let device = indices.device().clone();
    Tensor::from_vec(packed, (n, packed_dim), &Device::Cpu)?.to_device(&device)
}

/// Unpack bit-packed indices to U8 on the current device.
fn unpack_indices_on_device(
    packed: &Tensor,
    n: usize,
    head_dim: usize,
    bits: u8,
) -> Result<Tensor> {
    let device = packed.device().clone();
    let packed_cpu: Vec<u8> = packed.to_device(&Device::Cpu)?.flatten_all()?.to_vec1()?;
    let count = n * head_dim;
    let unpacked = match bits {
        2 => crate::packed::unpack_indices_2bit(&packed_cpu, count),
        3 => crate::packed::unpack_indices_3bit(&packed_cpu, count),
        4 => crate::packed::unpack_indices_4bit(&packed_cpu, count),
        _ => return Err(cache_err(format!("unsupported bits: {bits}"))),
    };
    Tensor::from_vec(unpacked, (n, head_dim), &Device::Cpu)?.to_device(&device)
}
