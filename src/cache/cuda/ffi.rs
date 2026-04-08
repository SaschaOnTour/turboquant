//! FFI declarations for TurboQuant CUDA kernels.

#[cfg(feature = "cuda")]
use std::ffi::c_int;

#[cfg(feature = "cuda")]
type CUstream = *const std::ffi::c_void;

#[cfg(feature = "cuda")]
extern "C" {
    pub fn tq_dequant_batch(
        packed_indices: *const u8,
        scales: *const u16,
        codebook: *const f32,
        sign_pattern: *const f32,
        output: *mut f32,
        num_blocks: c_int,
        block_size: c_int,
        bits: c_int,
        bytes_per_block: c_int,
        stream: CUstream,
    );

    pub fn tq_quant_maxnorm_batch(
        rotated_input: *const f32,
        boundaries: *const f32,
        packed_out: *mut u8,
        scales_out: *mut u16,
        num_blocks: c_int,
        block_size: c_int,
        bits: c_int,
        num_boundaries: c_int,
        bytes_per_block: c_int,
        outer_centroid: f32,
        scale_sign: f32,
        stream: CUstream,
    );

    #[allow(dead_code)] // Used later for GPU-side packing
    pub fn tq_pack_indices(
        indices: *const u8,
        packed_out: *mut u8,
        num_vectors: c_int,
        block_size: c_int,
        bits: c_int,
        bytes_per_block: c_int,
        stream: CUstream,
    );

    pub fn tq_fused_attention(
        q: *const f32,
        k_indices: *const u8,
        k_scales: *const u16,
        v_indices: *const u8,
        v_scales: *const u16,
        codebook: *const f32,
        sign_pattern: *const f32,
        output: *mut f32,
        partial_out: *mut f32,
        partial_max: *mut f32,
        partial_sum: *mut f32,
        // QJL correction (pass null + 0 when disabled)
        qjl_signs: *const u8,
        qjl_residual_norms: *const u16,
        rq: *const f32,
        qjl_enabled: c_int,
        num_attention_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        kv_len: c_int,
        kv_stride: c_int,
        packed_dim: c_int,
        num_qblocks: c_int,
        bits: c_int,
        softmax_scale: f32,
        num_partitions: c_int,
        stream: CUstream,
    );
}
