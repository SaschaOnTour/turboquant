//! Block-level PolarQuant on Candle tensors.
//!
//! Operates on `[N, head_dim]` tensors, splitting each vector into
//! `QUANT_BLOCK_SIZE`-element blocks for independent quantization.
//!
//! CPU path uses turboquant-rs core functions (pure Rust).
//! GPU path (TODO: Phase 4c) will use CUDA kernels.

use candle_core::{DType, Device, Result, Tensor};

use super::cache_err;
use super::config::{QuantNormMode, QUANT_BLOCK_SIZE};
use super::precomputed::GpuPrecomputed;
use super::wht_tensor::butterfly_wht_inverse_cpu;

/// Quantize `[N, head_dim]` f32 input to packed indices + scales.
///
/// Returns `(indices: [N, packed_dim] U8, scales: [N, num_blocks] F16)`
/// where `packed_dim = head_dim * bits / 8` and `num_blocks = head_dim / QUANT_BLOCK_SIZE`.
pub fn polar_quantize(
    input: &Tensor,
    head_dim: usize,
    bits: u8,
    norm_mode: QuantNormMode,
    outlier_blocks: usize,
    pre: &GpuPrecomputed,
) -> Result<(Tensor, Tensor)> {
    let n = input.dims()[0];
    let num_blocks = head_dim / QUANT_BLOCK_SIZE;
    const BITS_PER_BYTE: usize = 8;
    let packed_dim = head_dim * bits as usize / BITS_PER_BYTE;

    // CUDA fast path: matmul WHT + fused quant-and-pack kernel
    #[cfg(feature = "cuda")]
    if input.device().is_cuda()
        && norm_mode == QuantNormMode::MaxNorm
        && outlier_blocks >= num_blocks
    {
        return cuda_quantize_fast(input, n, head_dim, bits, num_blocks, packed_dim, pre);
    }

    const MIN_NORM: f64 = 1e-10;

    // Reshape to [N * num_blocks, QUANT_BLOCK_SIZE]
    let blocked = input.reshape((n * num_blocks, QUANT_BLOCK_SIZE))?;

    // Normalize + rotate (mode-dependent order)
    let (rotated, safe_norm) = match norm_mode {
        QuantNormMode::L2Norm => {
            let norm = blocked
                .sqr()?
                .sum_keepdim(1)?
                .sqrt()?
                .clamp(MIN_NORM, f64::MAX)?;
            let normalized = blocked.broadcast_div(&norm)?;
            let rotated = normalized.matmul(&pre.rotation_fwd)?;
            (rotated, norm)
        }
        QuantNormMode::MaxNorm => {
            let rotated_raw = blocked.matmul(&pre.rotation_fwd)?;
            let outer_c = pre.outlier_outer_centroid;
            let amax = rotated_raw.abs()?.max_keepdim(1)?;
            let scale = (amax / outer_c)?.clamp(MIN_NORM, f64::MAX)?;
            let rotated = rotated_raw.broadcast_div(&scale)?;
            (rotated, scale)
        }
    };

    // Outlier assignment: first `outlier_blocks` blocks use higher-bit codebook
    let effective_outlier = outlier_blocks.min(num_blocks);
    let outlier_rows = n * effective_outlier;
    let normal_start = outlier_rows;
    let normal_rows = n * num_blocks - outlier_rows;

    // Bucketize outlier blocks
    let idx_out = if outlier_rows > 0 {
        let n_ob = pre.outlier_boundaries.elem_count();
        let ob_exp = pre.outlier_boundaries.reshape((1, 1, n_ob))?;
        let rotated_out = rotated.narrow(0, 0, outlier_rows)?;
        Some(
            rotated_out
                .unsqueeze(2)?
                .broadcast_gt(&ob_exp)?
                .to_dtype(DType::U8)?
                .sum_keepdim(2)?
                .squeeze(2)?,
        )
    } else {
        None
    };

    // Bucketize normal blocks
    let idx_norm = if normal_rows > 0 {
        let n_nb = pre.boundaries.elem_count();
        let nb_exp = pre.boundaries.reshape((1, 1, n_nb))?;
        let rotated_norm = rotated.narrow(0, normal_start, normal_rows)?;
        Some(
            rotated_norm
                .unsqueeze(2)?
                .broadcast_gt(&nb_exp)?
                .to_dtype(DType::U8)?
                .sum_keepdim(2)?
                .squeeze(2)?,
        )
    } else {
        None
    };

    // Concatenate indices
    let indices = match (idx_out, idx_norm) {
        (Some(o), Some(n_idx)) => Tensor::cat(&[&o, &n_idx], 0)?,
        (Some(o), None) => o,
        (None, Some(n_idx)) => n_idx,
        (None, None) => return Err(cache_err("no blocks to quantize")),
    };

    let indices = indices.reshape((n, head_dim))?;
    let scales = safe_norm.reshape((n, num_blocks))?;

    // Mark outlier blocks with negative scale
    let scales = scales.broadcast_mul(&pre.scale_sign_tensor)?;
    let scales = scales.to_dtype(DType::F16)?;

    // Pack indices from U8 to bit-packed
    let packed_indices = pack_indices_cpu(&indices, n, packed_dim, bits)?;

    Ok((packed_indices, scales))
}

/// Dequantize packed indices + scales back to `[N, head_dim]` f32.
pub fn polar_dequantize(
    indices: &Tensor,
    scales: &Tensor,
    head_dim: usize,
    bits: u8,
    outlier_blocks: usize,
    pre: &GpuPrecomputed,
) -> Result<Tensor> {
    let n = indices.dims()[0];
    let num_blocks = head_dim / QUANT_BLOCK_SIZE;

    // CUDA fast path: fused unpack + codebook + WHT + scale kernel
    #[cfg(feature = "cuda")]
    if indices.device().is_cuda() && outlier_blocks >= num_blocks {
        return cuda_dequantize_fast(indices, scales, n, head_dim, bits, num_blocks, pre);
    }

    // CPU fallback: unpack + tensor ops
    let indices_unpacked = unpack_indices_on_device(indices, n, head_dim, bits)?;

    // Reshape to blocks
    let indices_blocked = indices_unpacked.reshape((n * num_blocks, QUANT_BLOCK_SIZE))?;
    let scales_blocked = scales.to_dtype(DType::F32)?.reshape((n * num_blocks, 1))?;
    let abs_scales = scales_blocked.abs()?;
    let indices_flat = indices_blocked.flatten_all()?.to_dtype(DType::U32)?;

    // Codebook lookup (select based on outlier config)
    let dequant = codebook_lookup(&indices_flat, &scales_blocked, outlier_blocks, num_blocks, n, pre)?;

    // Inverse WHT rotation
    let reconstructed = if indices.device().is_cpu() {
        butterfly_wht_inverse_cpu(&dequant, &pre.rotation_fwd, QUANT_BLOCK_SIZE)?
    } else {
        dequant.matmul(&pre.rotation_inv)?
    };

    // Re-scale and reshape
    reconstructed.broadcast_mul(&abs_scales)?.reshape((n, head_dim))
}

/// Codebook lookup with outlier/normal block dispatch.
fn codebook_lookup(
    indices_flat: &Tensor,
    scales_blocked: &Tensor,
    outlier_blocks: usize,
    num_blocks: usize,
    n: usize,
    pre: &GpuPrecomputed,
) -> Result<Tensor> {
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

// ---------------------------------------------------------------------------
// CUDA fast paths
// ---------------------------------------------------------------------------

/// CUDA fast dequantize: fused unpack + codebook + scale + WHT inverse kernel.
#[cfg(feature = "cuda")]
fn cuda_dequantize_fast(
    indices: &Tensor,
    scales: &Tensor,
    n: usize,
    head_dim: usize,
    bits: u8,
    num_blocks: usize,
    pre: &GpuPrecomputed,
) -> Result<Tensor> {
    use candle_core::cuda::cudarc::driver::DevicePtr;
    use candle_core::{DType, Storage};

    let total_blocks = n * num_blocks;
    let bytes_per_block = QUANT_BLOCK_SIZE * bits as usize / 8;
    let device = indices.device().clone();

    // Codebook + sign pattern for the kernel
    let sqrt_bs = (QUANT_BLOCK_SIZE as f64).sqrt();
    let sign_pattern = (pre.rotation_fwd.narrow(0, 0, 1)? * sqrt_bs)?
        .squeeze(0)?
        .to_dtype(DType::F32)?
        .contiguous()?;

    let output = Tensor::zeros(total_blocks * QUANT_BLOCK_SIZE, DType::F32, &device)?;

    let indices_cont = indices.flatten_all()?.contiguous()?;
    let scales_cont = scales.flatten_all()?.to_dtype(DType::F16)?.contiguous()?;
    let codebook_cont = pre.outlier_centroids.contiguous()?;

    let Device::Cuda(dev) = &device else {
        candle_core::bail!("Expected CUDA device");
    };
    let stream = dev.cuda_stream().cu_stream() as *const std::ffi::c_void;

    {
        let Storage::Cuda(i_cuda) = &*indices_cont.storage_and_layout().0 else {
            candle_core::bail!("indices must be on CUDA");
        };
        let Storage::Cuda(s_cuda) = &*scales_cont.storage_and_layout().0 else {
            candle_core::bail!("scales must be on CUDA");
        };
        let Storage::Cuda(c_cuda) = &*codebook_cont.storage_and_layout().0 else {
            candle_core::bail!("codebook must be on CUDA");
        };
        let Storage::Cuda(sp_cuda) = &*sign_pattern.storage_and_layout().0 else {
            candle_core::bail!("sign_pattern must be on CUDA");
        };
        let Storage::Cuda(o_cuda) = &*output.storage_and_layout().0 else {
            candle_core::bail!("output must be on CUDA");
        };

        let i_slice = i_cuda.as_cuda_slice::<u8>()?;
        let s_slice = s_cuda.as_cuda_slice::<half::f16>()?;
        let c_slice = c_cuda.as_cuda_slice::<f32>()?;
        let sp_slice = sp_cuda.as_cuda_slice::<f32>()?;
        let o_slice = o_cuda.as_cuda_slice::<f32>()?;

        let (i_ptr, _g1) = i_slice.device_ptr(i_slice.stream());
        let (s_ptr, _g2) = s_slice.device_ptr(s_slice.stream());
        let (c_ptr, _g3) = c_slice.device_ptr(c_slice.stream());
        let (sp_ptr, _g4) = sp_slice.device_ptr(sp_slice.stream());
        let (o_ptr, _g5) = o_slice.device_ptr(o_slice.stream());

        unsafe {
            super::cuda::ffi::tq_dequant_batch(
                i_ptr as *const u8,
                s_ptr as *const u16,
                c_ptr as *const f32,
                sp_ptr as *const f32,
                o_ptr as *mut f32,
                total_blocks as i32,
                QUANT_BLOCK_SIZE as i32,
                bits as i32,
                bytes_per_block as i32,
                stream,
            );
        }
    }

    output.reshape((n, head_dim))
}

/// CUDA fast quantize: matmul WHT + fused amax→scale→quantize→pack kernel.
#[cfg(feature = "cuda")]
fn cuda_quantize_fast(
    input: &Tensor,
    n: usize,
    _head_dim: usize,
    bits: u8,
    num_blocks: usize,
    packed_dim: usize,
    pre: &GpuPrecomputed,
) -> Result<(Tensor, Tensor)> {
    use candle_core::cuda::cudarc::driver::DevicePtr;
    use candle_core::{DType, Storage};

    let total_blocks = n * num_blocks;
    let bytes_per_block = QUANT_BLOCK_SIZE * bits as usize / 8;
    let n_ob = pre.outlier_boundaries.elem_count();
    let device = input.device().clone();

    // WHT rotation via matmul (fast on GPU via cuBLAS)
    let blocked = input.reshape((total_blocks, QUANT_BLOCK_SIZE))?;
    let rotated = blocked.matmul(&pre.rotation_fwd)?;
    let rotated_flat = rotated.flatten_all()?.contiguous()?;
    let boundaries_cont = pre.outlier_boundaries.contiguous()?;

    // Allocate output
    let packed_flat = candle_core::Tensor::zeros(total_blocks * bytes_per_block, DType::U8, &device)?;
    let scales_flat = candle_core::Tensor::zeros(total_blocks, DType::F16, &device)?;

    let Device::Cuda(dev) = &device else {
        candle_core::bail!("Expected CUDA device");
    };
    let stream = dev.cuda_stream().cu_stream() as *const std::ffi::c_void;

    {
        let Storage::Cuda(r_cuda) = &*rotated_flat.storage_and_layout().0 else {
            candle_core::bail!("rotated must be on CUDA");
        };
        let Storage::Cuda(b_cuda) = &*boundaries_cont.storage_and_layout().0 else {
            candle_core::bail!("boundaries must be on CUDA");
        };
        let Storage::Cuda(p_cuda) = &*packed_flat.storage_and_layout().0 else {
            candle_core::bail!("packed must be on CUDA");
        };
        let Storage::Cuda(s_cuda) = &*scales_flat.storage_and_layout().0 else {
            candle_core::bail!("scales must be on CUDA");
        };

        let r_slice = r_cuda.as_cuda_slice::<f32>()?;
        let b_slice = b_cuda.as_cuda_slice::<f32>()?;
        let p_slice = p_cuda.as_cuda_slice::<u8>()?;
        let s_slice = s_cuda.as_cuda_slice::<half::f16>()?;

        let (r_ptr, _g1) = r_slice.device_ptr(r_slice.stream());
        let (b_ptr, _g2) = b_slice.device_ptr(b_slice.stream());
        let (p_ptr, _g3) = p_slice.device_ptr(p_slice.stream());
        let (s_ptr, _g4) = s_slice.device_ptr(s_slice.stream());

        unsafe {
            super::cuda::ffi::tq_quant_maxnorm_batch(
                r_ptr as *const f32,
                b_ptr as *const f32,
                p_ptr as *mut u8,
                s_ptr as *mut u16,
                total_blocks as i32,
                QUANT_BLOCK_SIZE as i32,
                bits as i32,
                n_ob as i32,
                bytes_per_block as i32,
                pre.outlier_outer_centroid as f32,
                -1.0, // all outlier blocks -> negative scale
                stream,
            );
        }
    }

    let packed_indices = packed_flat.reshape((n, packed_dim))?;
    let scales = scales_flat.reshape((n, num_blocks))?;
    Ok((packed_indices, scales))
}
