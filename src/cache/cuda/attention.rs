//! CUDA fused attention kernel wrapper.
//!
//! Launches the partitioned fused attention kernel that reads directly
//! from the compressed KV-cache (no full dequantization needed).

#![cfg(feature = "cuda")]

use candle_core::cuda::cudarc::driver::DevicePtr;
use candle_core::{DType, Device, Result, Storage, Tensor};

use super::ffi;

const PARTITION_SIZE: usize = 512;

/// Parameters for the fused attention CUDA kernel.
pub struct FusedAttentionParams<'a> {
    pub q: &'a Tensor,
    pub k_indices: &'a Tensor,
    pub k_scales: &'a Tensor,
    pub v_indices: &'a Tensor,
    pub v_scales: &'a Tensor,
    pub codebook: &'a Tensor,
    pub sign_pattern: &'a Tensor,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub kv_len: usize,
    pub kv_stride: usize,
    pub packed_dim: usize,
    pub num_qblocks: usize,
    pub bits: usize,
    pub softmax_scale: f32,
    pub device: &'a Device,
}

/// Launch the TurboQuant fused attention kernel on GPU.
///
/// Two-phase: partitioned attention + reduce.
/// Returns attention output `[num_attention_heads, head_dim]`.
// qual:allow(TQ-003) — CUDA-only, tested via mistral.rs integration tests
pub fn fused_attention(p: &FusedAttentionParams<'_>) -> Result<Tensor> {
    let FusedAttentionParams {
        q,
        k_indices,
        k_scales,
        v_indices,
        v_scales,
        codebook,
        sign_pattern,
        num_attention_heads,
        num_kv_heads,
        head_dim,
        kv_len,
        kv_stride,
        packed_dim,
        num_qblocks,
        bits,
        softmax_scale,
        device,
    } = p;
    let Device::Cuda(dev) = device else {
        candle_core::bail!("fused_attention requires CUDA device");
    };

    let output = Tensor::zeros((*num_attention_heads, *head_dim), DType::F32, device)?;
    if *kv_len == 0 || *num_kv_heads == 0 {
        return Ok(output);
    }

    let num_partitions = (*kv_len + PARTITION_SIZE - 1) / PARTITION_SIZE;
    let partial_out = Tensor::zeros(
        (*num_attention_heads * num_partitions, *head_dim),
        DType::F32,
        device,
    )?;
    let partial_max = Tensor::zeros((*num_attention_heads * num_partitions,), DType::F32, device)?;
    let partial_sum = Tensor::zeros((*num_attention_heads * num_partitions,), DType::F32, device)?;

    let stream = dev.cuda_stream().cu_stream() as _;

    {
        let Storage::Cuda(q_cuda) = &*q.storage_and_layout().0 else {
            candle_core::bail!("q must be on CUDA");
        };
        let Storage::Cuda(ki_cuda) = &*k_indices.storage_and_layout().0 else {
            candle_core::bail!("k_indices must be on CUDA");
        };
        let Storage::Cuda(ks_cuda) = &*k_scales.storage_and_layout().0 else {
            candle_core::bail!("k_scales must be on CUDA");
        };
        let Storage::Cuda(vi_cuda) = &*v_indices.storage_and_layout().0 else {
            candle_core::bail!("v_indices must be on CUDA");
        };
        let Storage::Cuda(vs_cuda) = &*v_scales.storage_and_layout().0 else {
            candle_core::bail!("v_scales must be on CUDA");
        };
        let Storage::Cuda(cb_cuda) = &*codebook.storage_and_layout().0 else {
            candle_core::bail!("codebook must be on CUDA");
        };
        let Storage::Cuda(sp_cuda) = &*sign_pattern.storage_and_layout().0 else {
            candle_core::bail!("sign_pattern must be on CUDA");
        };
        let Storage::Cuda(out_cuda) = &*output.storage_and_layout().0 else {
            candle_core::bail!("output must be on CUDA");
        };
        let Storage::Cuda(po_cuda) = &*partial_out.storage_and_layout().0 else {
            candle_core::bail!("partial_out must be on CUDA");
        };
        let Storage::Cuda(pm_cuda) = &*partial_max.storage_and_layout().0 else {
            candle_core::bail!("partial_max must be on CUDA");
        };
        let Storage::Cuda(ps_cuda) = &*partial_sum.storage_and_layout().0 else {
            candle_core::bail!("partial_sum must be on CUDA");
        };

        let q_slice = q_cuda.as_cuda_slice::<f32>()?;
        let ki_slice = ki_cuda.as_cuda_slice::<u8>()?;
        let ks_slice = ks_cuda.as_cuda_slice::<half::f16>()?;
        let vi_slice = vi_cuda.as_cuda_slice::<u8>()?;
        let vs_slice = vs_cuda.as_cuda_slice::<half::f16>()?;
        let cb_slice = cb_cuda.as_cuda_slice::<f32>()?;
        let sp_slice = sp_cuda.as_cuda_slice::<f32>()?;
        let out_slice = out_cuda.as_cuda_slice::<f32>()?;
        let po_slice = po_cuda.as_cuda_slice::<f32>()?;
        let pm_slice = pm_cuda.as_cuda_slice::<f32>()?;
        let ps_slice = ps_cuda.as_cuda_slice::<f32>()?;

        let (q_ptr, _g1) = q_slice.device_ptr(q_slice.stream());
        let (ki_ptr, _g2) = ki_slice.device_ptr(ki_slice.stream());
        let (ks_ptr, _g3) = ks_slice.device_ptr(ks_slice.stream());
        let (vi_ptr, _g4) = vi_slice.device_ptr(vi_slice.stream());
        let (vs_ptr, _g5) = vs_slice.device_ptr(vs_slice.stream());
        let (cb_ptr, _g6) = cb_slice.device_ptr(cb_slice.stream());
        let (sp_ptr, _g7) = sp_slice.device_ptr(sp_slice.stream());
        let (out_ptr, _g8) = out_slice.device_ptr(out_slice.stream());
        let (po_ptr, _g9) = po_slice.device_ptr(po_slice.stream());
        let (pm_ptr, _g10) = pm_slice.device_ptr(pm_slice.stream());
        let (ps_ptr, _g11) = ps_slice.device_ptr(ps_slice.stream());

        unsafe {
            ffi::tq_fused_attention(
                q_ptr as *const f32,
                ki_ptr as *const u8,
                ks_ptr as *const u16,
                vi_ptr as *const u8,
                vs_ptr as *const u16,
                cb_ptr as *const f32,
                sp_ptr as *const f32,
                out_ptr as *mut f32,
                po_ptr as *mut f32,
                pm_ptr as *mut f32,
                ps_ptr as *mut f32,
                // No QJL for PQO
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0, // qjl_enabled = false
                *num_attention_heads as i32,
                *num_kv_heads as i32,
                *head_dim as i32,
                *kv_len as i32,
                *kv_stride as i32,
                *packed_dim as i32,
                *num_qblocks as i32,
                *bits as i32,
                *softmax_scale,
                num_partitions as i32,
                stream,
            );
        }
    }

    Ok(output)
}
