//! Cache-level roundtrip NMSE tests on the CPU path.
//!
//! Feeds deterministic K/V through each cache variant and measures the
//! normalized MSE between the dequantized output and the original input.
//! This covers the full Rotate → Normalize → Quantize → Pack → Unpack →
//! Dequantize pipeline at the cache boundary, not just the core packed
//! functions (which `mse_polar_tests` already exercises).
//!
//! Expected NMSE ranges are loose enough to avoid flakiness but tight
//! enough to surface packing/codebook/rotation regressions.

#![cfg(feature = "candle")]

use candle_core::{DType, Device, Result, Tensor};
use mistralrs_kv_cache::CompressedKVCache;
use turboquant::cache::config::QuantNormMode;
use turboquant::cache::{CacheConfig, PqoCache, TqCache};
use turboquant::test_utils::{make_q, random_normal_vec};

const HEAD_DIM: usize = 128;
const NUM_KV_HEADS: usize = 4;
const NUM_ATTN_HEADS: usize = NUM_KV_HEADS * 2;
const LAYER: usize = 0;

/// Two-phase prefill: enough tokens to keep per-block MSE statistically stable.
const PREFILL_A_LEN: usize = 64;
const PREFILL_B_LEN: usize = 16;
const SEED_A: u32 = 101;
const SEED_B: u32 = 202;

/// NMSE ranges are calibrated on the deterministic Gaussian input above —
/// ±30 % below, +50 % above the observed value, wide enough for float-reduction
/// variance across platforms but tight enough to catch a 2× regression or a
/// "no quantization" bug. Note: cache-level NMSE is ~10× higher than vector-
/// level polar NMSE (see `mse_polar_tests`) because the cache does block-wise
/// quantization (32-element blocks) which amortizes rotation less efficiently.
struct NmseRange {
    min: f64,
    max: f64,
}

/// PQ3 / TQ3 — 2-bit polar (normal codebook only); observed ~0.32.
const PQ3_TQ3_NMSE: NmseRange = NmseRange {
    min: 0.22,
    max: 0.48,
};
/// PQO3 / TQ4 — 3-bit (outlier codebook for PQO3, polar for TQ4); observed ~0.031.
const PQO3_TQ4_NMSE: NmseRange = NmseRange {
    min: 0.022,
    max: 0.047,
};
/// PQO4 — 4-bit outlier codebook; observed ~0.008.
const PQO4_NMSE: NmseRange = NmseRange {
    min: 0.005,
    max: 0.012,
};

/// Gaussian K/V — each element iid N(0,1), matching the input distribution
/// used by `mse_polar_tests` so observed NMSE aligns with those ranges.
/// K and V use disjoint seed spaces so they are statistically independent.
fn gaussian_kv(seq_len: usize, seed_base: u64, device: &Device) -> Result<(Tensor, Tensor)> {
    let total = NUM_KV_HEADS * seq_len * HEAD_DIM;
    let k_data = random_normal_vec(total, seed_base);
    let v_data = random_normal_vec(total, seed_base.wrapping_add(V_SEED_STRIDE));
    let shape = (1, NUM_KV_HEADS, seq_len, HEAD_DIM);
    Ok((
        Tensor::from_vec(k_data, shape, device)?,
        Tensor::from_vec(v_data, shape, device)?,
    ))
}

/// Offset between K and V random-seed ranges so they draw disjoint streams.
const V_SEED_STRIDE: u64 = 1_000_000;

fn cache_config(bits: u8, outlier_blocks: usize) -> CacheConfig {
    CacheConfig {
        bits,
        head_dim: HEAD_DIM,
        num_kv_heads: NUM_KV_HEADS,
        num_layers: 1,
        norm_mode: QuantNormMode::MaxNorm,
        outlier_blocks,
    }
}

fn nmse(reconstructed: &Tensor, original: &Tensor) -> Result<f64> {
    let orig = original.to_dtype(DType::F32)?;
    let recon = reconstructed.to_dtype(DType::F32)?;
    let err_sq = (recon - &orig)?.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
    let norm_sq = orig.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
    Ok(err_sq / norm_sq)
}

/// Drives a two-prefill sequence and returns `(k_nmse, v_nmse)` against the
/// original concatenated reference. The first prefill echoes the input
/// (never dequantizes), the second returns the full dequantized cache —
/// that's what we compare against.
pub(crate) fn roundtrip_nmse(cache: &dyn CompressedKVCache, device: &Device) -> Result<(f64, f64)> {
    let (k_a, v_a) = gaussian_kv(PREFILL_A_LEN, SEED_A as u64, device)?;
    let (k_b, v_b) = gaussian_kv(PREFILL_B_LEN, SEED_B as u64, device)?;
    let q_a = make_q(PREFILL_A_LEN, NUM_ATTN_HEADS, HEAD_DIM).to_device(device)?;
    let q_b = make_q(PREFILL_B_LEN, NUM_ATTN_HEADS, HEAD_DIM).to_device(device)?;

    cache.prefill(LAYER, &k_a, &v_a, &q_a)?;
    let result = cache.prefill(LAYER, &k_b, &v_b, &q_b)?;

    let k_ref = Tensor::cat(&[&k_a, &k_b], 2)?;
    let v_ref = Tensor::cat(&[&v_a, &v_b], 2)?;

    Ok((nmse(&result.k, &k_ref)?, nmse(&result.v, &v_ref)?))
}

pub(crate) fn assert_nmse_in_range(label: &str, k: f64, v: f64, range: &NmseRange) {
    eprintln!(
        "{label} K NMSE: {k:.6}, V NMSE: {v:.6} (range: [{min}, {max}])",
        min = range.min,
        max = range.max
    );
    assert!(
        (range.min..=range.max).contains(&k),
        "{label} K NMSE {k:.6} outside [{}, {}]",
        range.min,
        range.max
    );
    assert!(
        (range.min..=range.max).contains(&v),
        "{label} V NMSE {v:.6} outside [{}, {}]",
        range.min,
        range.max
    );
}

#[test]
fn pq3_cpu_roundtrip_nmse_in_range() -> Result<()> {
    let cache = PqoCache::new(cache_config(3, 0))?;
    let (k, v) = roundtrip_nmse(&cache, &Device::Cpu)?;
    assert_nmse_in_range("PQ3", k, v, &PQ3_TQ3_NMSE);
    Ok(())
}

#[test]
fn tq3_cpu_roundtrip_nmse_in_range() -> Result<()> {
    let cache = TqCache::new(cache_config(3, 0))?;
    let (k, v) = roundtrip_nmse(&cache, &Device::Cpu)?;
    assert_nmse_in_range("TQ3", k, v, &PQ3_TQ3_NMSE);
    Ok(())
}

#[test]
fn tq4_cpu_roundtrip_nmse_in_range() -> Result<()> {
    let cache = TqCache::new(cache_config(4, 0))?;
    let (k, v) = roundtrip_nmse(&cache, &Device::Cpu)?;
    assert_nmse_in_range("TQ4", k, v, &PQO3_TQ4_NMSE);
    Ok(())
}

#[test]
fn pqo3_cpu_roundtrip_nmse_in_range() -> Result<()> {
    let cache = PqoCache::new(cache_config(3, usize::MAX))?;
    let (k, v) = roundtrip_nmse(&cache, &Device::Cpu)?;
    assert_nmse_in_range("PQO3", k, v, &PQO3_TQ4_NMSE);
    Ok(())
}

#[test]
fn pqo4_cpu_roundtrip_nmse_in_range() -> Result<()> {
    let cache = PqoCache::new(cache_config(4, usize::MAX))?;
    let (k, v) = roundtrip_nmse(&cache, &Device::Cpu)?;
    assert_nmse_in_range("PQO4", k, v, &PQO4_NMSE);
    Ok(())
}
