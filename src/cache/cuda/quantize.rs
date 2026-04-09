//! CUDA fast-path quantize/dequantize kernel wrappers.
//!
//! Moved from `quantize_tensor.rs` to reduce module length.
//! Functions use inline CUDA storage extraction (no shared helper)
//! to keep each function's variable names distinct.

#![cfg(feature = "cuda")]

use candle_core::cuda::cudarc::driver::DevicePtr;
use candle_core::{DType, Device, Result, Storage, Tensor};

use super::ffi;
use crate::cache::config::{BITS_PER_BYTE, QUANT_BLOCK_SIZE};
use crate::cache::quantize_tensor::QuantConfig;

/// CUDA fast dequantize: fused unpack + codebook + scale + WHT inverse kernel.
// qual:allow(complexity) — linear FFI boilerplate, unsafe required for CUDA kernel call
pub fn cuda_dequantize_fast(
    indices: &Tensor,
    scales: &Tensor,
    n: usize,
    config: &QuantConfig<'_>,
) -> Result<Tensor> {
    let head_dim = config.head_dim;
    let bits = config.bits;
    let num_blocks = config.num_blocks();
    let pre = config.pre;
    let total_blocks = n * num_blocks;
    let bytes_per_block = QUANT_BLOCK_SIZE * bits as usize / BITS_PER_BYTE;
    let device = indices.device().clone();

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
        let Storage::Cuda(i_s) = &*indices_cont.storage_and_layout().0 else {
            candle_core::bail!("CUDA")
        };
        let Storage::Cuda(s_s) = &*scales_cont.storage_and_layout().0 else {
            candle_core::bail!("CUDA")
        };
        let Storage::Cuda(c_s) = &*codebook_cont.storage_and_layout().0 else {
            candle_core::bail!("CUDA")
        };
        let Storage::Cuda(sp_s) = &*sign_pattern.storage_and_layout().0 else {
            candle_core::bail!("CUDA")
        };
        let Storage::Cuda(o_s) = &*output.storage_and_layout().0 else {
            candle_core::bail!("CUDA")
        };
        let i_sl = i_s.as_cuda_slice::<u8>()?;
        let s_sl = s_s.as_cuda_slice::<half::f16>()?;
        let c_sl = c_s.as_cuda_slice::<f32>()?;
        let sp_sl = sp_s.as_cuda_slice::<f32>()?;
        let o_sl = o_s.as_cuda_slice::<f32>()?;
        let (i_ptr, _g1) = i_sl.device_ptr(i_sl.stream());
        let (s_ptr, _g2) = s_sl.device_ptr(s_sl.stream());
        let (c_ptr, _g3) = c_sl.device_ptr(c_sl.stream());
        let (sp_ptr, _g4) = sp_sl.device_ptr(sp_sl.stream());
        let (o_ptr, _g5) = o_sl.device_ptr(o_sl.stream());

        // qual:allow(complexity) — FFI call to CUDA kernel requires unsafe
        unsafe {
            ffi::tq_dequant_batch(
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
// qual:allow(complexity) — linear FFI boilerplate, unsafe required for CUDA kernel call
pub fn cuda_quantize_fast(
    input: &Tensor,
    n: usize,
    config: &QuantConfig<'_>,
) -> Result<(Tensor, Tensor)> {
    let bits = config.bits;
    let num_blocks = config.num_blocks();
    let packed_dim = config.packed_dim();
    let pre = config.pre;
    let total_blocks = n * num_blocks;
    let bytes_per_block = QUANT_BLOCK_SIZE * bits as usize / BITS_PER_BYTE;
    let n_ob = pre.outlier_boundaries.elem_count();
    let device = input.device().clone();

    let blocked = input.reshape((total_blocks, QUANT_BLOCK_SIZE))?;
    let rotated = blocked.matmul(&pre.rotation_fwd)?;
    let rotated_flat = rotated.flatten_all()?.contiguous()?;
    let boundaries_cont = pre.outlier_boundaries.contiguous()?;

    let packed_flat = Tensor::zeros(total_blocks * bytes_per_block, DType::U8, &device)?;
    let scales_flat = Tensor::zeros(total_blocks, DType::F16, &device)?;

    let Device::Cuda(dev) = &device else {
        candle_core::bail!("Expected CUDA device");
    };
    let stream = dev.cuda_stream().cu_stream() as *const std::ffi::c_void;

    {
        let Storage::Cuda(r_s) = &*rotated_flat.storage_and_layout().0 else {
            candle_core::bail!("CUDA")
        };
        let Storage::Cuda(b_s) = &*boundaries_cont.storage_and_layout().0 else {
            candle_core::bail!("CUDA")
        };
        let Storage::Cuda(p_s) = &*packed_flat.storage_and_layout().0 else {
            candle_core::bail!("CUDA")
        };
        let Storage::Cuda(s_s) = &*scales_flat.storage_and_layout().0 else {
            candle_core::bail!("CUDA")
        };
        let r_sl = r_s.as_cuda_slice::<f32>()?;
        let b_sl = b_s.as_cuda_slice::<f32>()?;
        let p_sl = p_s.as_cuda_slice::<u8>()?;
        let s_sl = s_s.as_cuda_slice::<half::f16>()?;
        let (r_ptr, _g1) = r_sl.device_ptr(r_sl.stream());
        let (b_ptr, _g2) = b_sl.device_ptr(b_sl.stream());
        let (p_ptr, _g3) = p_sl.device_ptr(p_sl.stream());
        let (s_ptr, _g4) = s_sl.device_ptr(s_sl.stream());

        // qual:allow(complexity) — FFI call to CUDA kernel requires unsafe
        unsafe {
            ffi::tq_quant_maxnorm_batch(
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
                -1.0,
                stream,
            );
        }
    }

    let packed_indices = packed_flat.reshape((n, packed_dim))?;
    let scales = scales_flat.reshape((n, num_blocks))?;
    Ok((packed_indices, scales))
}
