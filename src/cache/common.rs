//! Shared helpers for PqoCache and TqCache implementations.

use candle_core::{DType, Result, Tensor};
use mistralrs_kv_cache::DequantResult;

use super::cache_err;
use super::config::{CacheConfig, QUANT_BLOCK_SIZE};
use super::precomputed::GpuPrecomputed;
use super::quantize_tensor::{polar_dequantize, QuantConfig};
use super::storage::{LayerStorage, StorageMetadata};

/// Validate `config.head_dim` is divisible by `QUANT_BLOCK_SIZE` and return
/// the derived read-only `StorageMetadata`.
///
/// Used by both `PqoCache::new` and `TqCache::new` to share the divisibility
/// check and metadata construction.
pub(crate) fn validate_and_make_metadata(config: &CacheConfig) -> Result<StorageMetadata> {
    if config.head_dim % QUANT_BLOCK_SIZE != 0 {
        candle_core::bail!(
            "head_dim ({}) must be divisible by QUANT_BLOCK_SIZE ({QUANT_BLOCK_SIZE}). \
             Models with head_dim={} are not supported by TurboQuant compression.",
            config.head_dim,
            config.head_dim
        );
    }
    Ok(StorageMetadata {
        num_kv_heads: config.num_kv_heads,
        head_dim: config.head_dim,
        bits: config.bits,
    })
}

/// Dequantize the full compressed cache for a single layer slot.
///
/// Shared implementation used by both `PqoCache` and `TqCache`.
// qual:allow(TQ-003) — tested via cache_pqo_tests + cache_storage_tests integration tests
pub(crate) fn dequantize_full_impl(
    layer: &LayerStorage,
    metadata: &StorageMetadata,
    config: &QuantConfig<'_>,
    orig_dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let total_seq = layer.seq_len();
    let head_dim = metadata.head_dim;
    let num_kv_heads = metadata.num_kv_heads;
    let packed_dim = metadata.packed_dim();
    let num_blocks = metadata.num_blocks();

    let bufs = layer
        .buffers()
        .ok_or_else(|| cache_err("layer buffers not initialized"))?;

    let all_ki = bufs
        .k_indices
        .narrow(1, 0, total_seq)?
        .reshape((num_kv_heads * total_seq, packed_dim))?;
    let all_ks = bufs
        .k_scales
        .narrow(1, 0, total_seq)?
        .reshape((num_kv_heads * total_seq, num_blocks))?;
    let all_vi = bufs
        .v_indices
        .narrow(1, 0, total_seq)?
        .reshape((num_kv_heads * total_seq, packed_dim))?;
    let all_vs = bufs
        .v_scales
        .narrow(1, 0, total_seq)?
        .reshape((num_kv_heads * total_seq, num_blocks))?;

    let full_k = polar_dequantize(&all_ki, &all_ks, config)?
        .reshape((1, num_kv_heads, total_seq, head_dim))?
        .to_dtype(orig_dtype)?;
    let full_v = polar_dequantize(&all_vi, &all_vs, config)?
        .reshape((1, num_kv_heads, total_seq, head_dim))?
        .to_dtype(orig_dtype)?;

    Ok((full_k, full_v))
}

/// Build a [`QuantConfig`] from precomputed tensors and cache configuration.
pub(crate) fn make_quant_config<'a>(
    precomputed: &'a GpuPrecomputed,
    config: &CacheConfig,
) -> QuantConfig<'a> {
    QuantConfig {
        head_dim: config.head_dim,
        bits: config.bits,
        outlier_blocks: config.outlier_blocks,
        pre: precomputed,
    }
}

/// Flatten K/V tensors from `[1, heads, seq, dim]` to `[heads*seq, dim]` as f32.
pub(crate) fn flatten_kv(
    k: &Tensor,
    v: &Tensor,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<(Tensor, Tensor)> {
    let new_seq_len = k.dims()[2];
    let k_flat = k
        .squeeze(0)?
        .to_dtype(DType::F32)?
        .reshape((num_kv_heads * new_seq_len, head_dim))?;
    let v_flat = v
        .squeeze(0)?
        .to_dtype(DType::F32)?
        .reshape((num_kv_heads * new_seq_len, head_dim))?;
    Ok((k_flat, v_flat))
}

/// Quantize a K/V pair using polar quantization.
///
/// Returns `(k_indices, k_scales, v_indices, v_scales)` in flat format.
pub(crate) fn quantize_kv_pair(
    k_flat: &Tensor,
    v_flat: &Tensor,
    norm_mode: super::config::QuantNormMode,
    qc: &super::quantize_tensor::QuantConfig<'_>,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let (k_idx, k_sc) = super::quantize_tensor::polar_quantize(k_flat, norm_mode, qc)?;
    let (v_idx, v_sc) = super::quantize_tensor::polar_quantize(v_flat, norm_mode, qc)?;
    Ok((k_idx, k_sc, v_idx, v_sc))
}

/// Create a `DequantResult` with no logit bias (PQO mode).
// qual:allow(TQ-003) — trivial constructor, tested through PqoCache integration tests
pub(crate) fn dequant_result(k: Tensor, v: Tensor) -> DequantResult {
    DequantResult {
        k,
        v,
        logit_bias: None,
    }
}
