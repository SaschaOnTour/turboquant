//! Shared helpers for PqoCache and TqCache implementations.

use candle_core::{DType, Result, Tensor};
use mistralrs_kv_cache::DequantResult;

use super::cache_err;
use super::config::CacheConfig;
use super::precomputed::GpuPrecomputed;
use super::quantize_tensor::{polar_dequantize, QuantConfig};
use super::storage::CompressedStorage;

/// Dequantize the full compressed cache for a layer.
///
/// Shared implementation used by both `PqoCache` and `TqCache`.
// qual:allow(TQ-003) — tested via cache_pqo_tests + cache_storage_tests integration tests
pub(crate) fn dequantize_full_impl(
    storage: &CompressedStorage,
    config: &QuantConfig<'_>,
    layer: usize,
    orig_dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let total_seq = storage.seq_len(layer);
    let head_dim = storage.head_dim;
    let num_kv_heads = storage.num_kv_heads;
    let packed_dim = storage.packed_dim();
    let num_blocks = storage.num_blocks();

    let ki = storage
        .k_indices(layer)
        .ok_or_else(|| cache_err("k_indices not initialized"))?;
    let ks = storage
        .k_scales(layer)
        .ok_or_else(|| cache_err("k_scales not initialized"))?;
    let vi = storage
        .v_indices(layer)
        .ok_or_else(|| cache_err("v_indices not initialized"))?;
    let vs = storage
        .v_scales(layer)
        .ok_or_else(|| cache_err("v_scales not initialized"))?;

    let all_ki = ki
        .narrow(1, 0, total_seq)?
        .reshape((num_kv_heads * total_seq, packed_dim))?;
    let all_ks = ks
        .narrow(1, 0, total_seq)?
        .reshape((num_kv_heads * total_seq, num_blocks))?;
    let all_vi = vi
        .narrow(1, 0, total_seq)?
        .reshape((num_kv_heads * total_seq, packed_dim))?;
    let all_vs = vs
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
    precomputed: &'a Option<GpuPrecomputed>,
    config: &CacheConfig,
) -> Result<QuantConfig<'a>> {
    let pre = precomputed
        .as_ref()
        .ok_or_else(|| cache_err("precomputed not initialized"))?;
    Ok(QuantConfig {
        head_dim: config.head_dim,
        bits: config.bits,
        outlier_blocks: config.outlier_blocks,
        pre,
    })
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
