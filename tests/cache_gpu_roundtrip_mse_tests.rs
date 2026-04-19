//! Cache-level roundtrip NMSE tests on the CUDA path.
//!
//! Mirrors `cache_cpu_roundtrip_mse_tests` but runs every cache variant
//! against CUDA-allocated K/V tensors. The PQO variants exercise the
//! `cuda_quantize_fast` GPU kernel; TQ/PQ fall back to the CPU algorithm
//! on GPU tensors but still verify that the device-agnostic pipeline is
//! numerically stable when data lives on the device.
//!
//! NMSE ranges are calibrated off CPU values plus a larger tolerance to
//! absorb float-reduction differences between CPU and GPU backends.

#![cfg(all(feature = "candle", feature = "cuda"))]

use candle_core::{DType, Device, Result, Tensor};
use mistralrs_kv_cache::CompressedKVCache;
use turboquant::cache::config::QuantNormMode;
use turboquant::cache::{CacheConfig, PqoCache, TqCache};
use turboquant::test_utils::{make_q, random_normal_vec};

const HEAD_DIM: usize = 128;
const NUM_KV_HEADS: usize = 4;
const NUM_ATTN_HEADS: usize = NUM_KV_HEADS * 2;
const LAYER: usize = 0;

const PREFILL_A_LEN: usize = 64;
const PREFILL_B_LEN: usize = 16;
const SEED_A: u64 = 101;
const SEED_B: u64 = 202;
const V_SEED_STRIDE: u64 = 1_000_000;

struct NmseRange {
    min: f64,
    max: f64,
}

/// PQ3 / TQ3 — 2-bit polar (normal codebook only); CPU observed ~0.32.
const PQ3_TQ3_NMSE: NmseRange = NmseRange {
    min: 0.20,
    max: 0.50,
};
/// PQO3 / TQ4 — 3-bit codebook; CPU observed ~0.031.
const PQO3_TQ4_NMSE: NmseRange = NmseRange {
    min: 0.020,
    max: 0.050,
};
/// PQO4 — 4-bit outlier codebook; CPU observed ~0.008.
const PQO4_NMSE: NmseRange = NmseRange {
    min: 0.004,
    max: 0.014,
};

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

fn roundtrip_nmse(cache: &dyn CompressedKVCache, device: &Device) -> Result<(f64, f64)> {
    let (k_a, v_a) = gaussian_kv(PREFILL_A_LEN, SEED_A, device)?;
    let (k_b, v_b) = gaussian_kv(PREFILL_B_LEN, SEED_B, device)?;
    let q_a = make_q(PREFILL_A_LEN, NUM_ATTN_HEADS, HEAD_DIM).to_device(device)?;
    let q_b = make_q(PREFILL_B_LEN, NUM_ATTN_HEADS, HEAD_DIM).to_device(device)?;

    cache.prefill(LAYER, &k_a, &v_a, &q_a)?;
    let result = cache.prefill(LAYER, &k_b, &v_b, &q_b)?;

    let k_ref = Tensor::cat(&[&k_a, &k_b], 2)?;
    let v_ref = Tensor::cat(&[&v_a, &v_b], 2)?;

    Ok((nmse(&result.k, &k_ref)?, nmse(&result.v, &v_ref)?))
}

fn assert_nmse_in_range(label: &str, k: f64, v: f64, range: &NmseRange) {
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

fn cuda_device() -> Device {
    Device::cuda_if_available(0).expect("CUDA device required")
}

#[test]
fn pq3_gpu_roundtrip_nmse_in_range() -> Result<()> {
    let cache = PqoCache::new(cache_config(3, 0))?;
    let (k, v) = roundtrip_nmse(&cache, &cuda_device())?;
    assert_nmse_in_range("PQ3", k, v, &PQ3_TQ3_NMSE);
    Ok(())
}

#[test]
fn tq3_gpu_roundtrip_nmse_in_range() -> Result<()> {
    let cache = TqCache::new(cache_config(3, 0))?;
    let (k, v) = roundtrip_nmse(&cache, &cuda_device())?;
    assert_nmse_in_range("TQ3", k, v, &PQ3_TQ3_NMSE);
    Ok(())
}

#[test]
fn tq4_gpu_roundtrip_nmse_in_range() -> Result<()> {
    let cache = TqCache::new(cache_config(4, 0))?;
    let (k, v) = roundtrip_nmse(&cache, &cuda_device())?;
    assert_nmse_in_range("TQ4", k, v, &PQO3_TQ4_NMSE);
    Ok(())
}

#[test]
fn pqo3_gpu_roundtrip_nmse_in_range() -> Result<()> {
    let cache = PqoCache::new(cache_config(3, usize::MAX))?;
    let (k, v) = roundtrip_nmse(&cache, &cuda_device())?;
    assert_nmse_in_range("PQO3", k, v, &PQO3_TQ4_NMSE);
    Ok(())
}

#[test]
fn pqo4_gpu_roundtrip_nmse_in_range() -> Result<()> {
    let cache = PqoCache::new(cache_config(4, usize::MAX))?;
    let (k, v) = roundtrip_nmse(&cache, &cuda_device())?;
    assert_nmse_in_range("PQO4", k, v, &PQO4_NMSE);
    Ok(())
}
