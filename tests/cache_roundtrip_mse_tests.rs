//! Cache-level roundtrip NMSE tests for the CPU and CUDA code paths.
//!
//! Feeds deterministic Gaussian K/V through each cache variant and measures
//! the normalized MSE between the dequantized output and the original input.
//! Covers the full Rotate → Normalize → Quantize → Pack → Unpack → Dequantize
//! pipeline at the cache boundary — not just the core packed functions that
//! `mse_polar_tests` exercises.
//!
//! Observed NMSE is deterministic for a given input seed, codebook, and
//! rotation; ranges are calibrated as `±30 %` below / `+50 %` above the
//! observed value — loose enough to absorb float-reduction variance across
//! backends, tight enough to flag a 2× quality regression or an accidental
//! "bit-identical pass-through" bug.
//!
//! PQO variants on the CUDA path exercise `cuda_quantize_fast` (the GPU
//! kernel); TQ/PQ on CUDA fall back to the CPU algorithm operating on
//! GPU tensors. All code paths share the same NMSE contract here.

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

const PREFILL_A_LEN: usize = 64;
const PREFILL_B_LEN: usize = 16;
const SEED_A: u64 = 101;
const SEED_B: u64 = 202;
/// Offset between K and V random-seed streams so they are statistically independent.
const V_SEED_STRIDE: u64 = 1_000_000;

struct NmseRange {
    min: f64,
    max: f64,
}

/// PQ3 / TQ3 — 2-bit polar (normal codebook only); CPU+GPU both observed ~0.32.
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

fn make_config(bits: u8, outlier_blocks: usize) -> CacheConfig {
    CacheConfig {
        bits,
        head_dim: HEAD_DIM,
        num_kv_heads: NUM_KV_HEADS,
        num_layers: 1,
        norm_mode: QuantNormMode::MaxNorm,
        outlier_blocks,
    }
}

/// i.i.d. N(0,1) K/V factory. K and V draw from disjoint seed ranges to avoid
/// accidental correlation.
// qual:allow(iosp) — test factory: minimal arithmetic to derive buffer sizes
// from declared dimensions before calling into `random_normal_vec` /
// `Tensor::from_vec`; splitting would only add ceremony.
fn gaussian_kv(seq_len: usize, seed_base: u64, device: &Device) -> Result<(Tensor, Tensor)> {
    let k_data = random_normal_vec(NUM_KV_HEADS * seq_len * HEAD_DIM, seed_base);
    let v_data = random_normal_vec(
        NUM_KV_HEADS * seq_len * HEAD_DIM,
        seed_base.wrapping_add(V_SEED_STRIDE),
    );
    let shape = (1, NUM_KV_HEADS, seq_len, HEAD_DIM);
    Ok((
        Tensor::from_vec(k_data, shape, device)?,
        Tensor::from_vec(v_data, shape, device)?,
    ))
}

fn compute_nmse(reconstructed: &Tensor, original: &Tensor) -> Result<f64> {
    let orig = original.to_dtype(DType::F32)?;
    let recon = reconstructed.to_dtype(DType::F32)?;
    let err_sq = (recon - &orig)?.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
    let norm_sq = orig.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
    Ok(err_sq / norm_sq)
}

/// Two-phase prefill: the first call echoes the input unchanged (never
/// dequantizes), the second returns the full dequantized `{A, B}` — that's
/// what we compare against the concatenated reference.
fn roundtrip_nmse(cache: &dyn CompressedKVCache, device: &Device) -> Result<(f64, f64)> {
    let (k_a, v_a) = gaussian_kv(PREFILL_A_LEN, SEED_A, device)?;
    let (k_b, v_b) = gaussian_kv(PREFILL_B_LEN, SEED_B, device)?;
    let q_a = make_q(PREFILL_A_LEN, NUM_ATTN_HEADS, HEAD_DIM).to_device(device)?;
    let q_b = make_q(PREFILL_B_LEN, NUM_ATTN_HEADS, HEAD_DIM).to_device(device)?;

    cache.prefill(LAYER, &k_a, &v_a, &q_a)?;
    let result = cache.prefill(LAYER, &k_b, &v_b, &q_b)?;

    let k_ref = Tensor::cat(&[&k_a, &k_b], 2)?;
    let v_ref = Tensor::cat(&[&v_a, &v_b], 2)?;

    Ok((
        compute_nmse(&result.k, &k_ref)?,
        compute_nmse(&result.v, &v_ref)?,
    ))
}

fn assert_in_range(label: &str, k: f64, v: f64, range: &NmseRange) {
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

// ---- CPU tests ---------------------------------------------------------

#[test]
fn pq3_cpu_roundtrip_nmse_in_range() -> Result<()> {
    let cache = PqoCache::new(make_config(3, 0))?;
    let (k, v) = roundtrip_nmse(&cache, &Device::Cpu)?;
    assert_in_range("PQ3 CPU", k, v, &PQ3_TQ3_NMSE);
    Ok(())
}

#[test]
fn tq3_cpu_roundtrip_nmse_in_range() -> Result<()> {
    let cache = TqCache::new(make_config(3, 0))?;
    let (k, v) = roundtrip_nmse(&cache, &Device::Cpu)?;
    assert_in_range("TQ3 CPU", k, v, &PQ3_TQ3_NMSE);
    Ok(())
}

#[test]
fn tq4_cpu_roundtrip_nmse_in_range() -> Result<()> {
    let cache = TqCache::new(make_config(4, 0))?;
    let (k, v) = roundtrip_nmse(&cache, &Device::Cpu)?;
    assert_in_range("TQ4 CPU", k, v, &PQO3_TQ4_NMSE);
    Ok(())
}

#[test]
fn pqo3_cpu_roundtrip_nmse_in_range() -> Result<()> {
    let cache = PqoCache::new(make_config(3, usize::MAX))?;
    let (k, v) = roundtrip_nmse(&cache, &Device::Cpu)?;
    assert_in_range("PQO3 CPU", k, v, &PQO3_TQ4_NMSE);
    Ok(())
}

#[test]
fn pqo4_cpu_roundtrip_nmse_in_range() -> Result<()> {
    let cache = PqoCache::new(make_config(4, usize::MAX))?;
    let (k, v) = roundtrip_nmse(&cache, &Device::Cpu)?;
    assert_in_range("PQO4 CPU", k, v, &PQO4_NMSE);
    Ok(())
}

// ---- CUDA tests --------------------------------------------------------

#[cfg(feature = "cuda")]
fn cuda_device() -> Device {
    Device::cuda_if_available(0).expect("CUDA device required")
}

#[cfg(feature = "cuda")]
#[test]
fn pq3_gpu_roundtrip_nmse_in_range() -> Result<()> {
    let cache = PqoCache::new(make_config(3, 0))?;
    let (k, v) = roundtrip_nmse(&cache, &cuda_device())?;
    assert_in_range("PQ3 GPU", k, v, &PQ3_TQ3_NMSE);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn tq3_gpu_roundtrip_nmse_in_range() -> Result<()> {
    let cache = TqCache::new(make_config(3, 0))?;
    let (k, v) = roundtrip_nmse(&cache, &cuda_device())?;
    assert_in_range("TQ3 GPU", k, v, &PQ3_TQ3_NMSE);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn tq4_gpu_roundtrip_nmse_in_range() -> Result<()> {
    let cache = TqCache::new(make_config(4, 0))?;
    let (k, v) = roundtrip_nmse(&cache, &cuda_device())?;
    assert_in_range("TQ4 GPU", k, v, &PQO3_TQ4_NMSE);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn pqo3_gpu_roundtrip_nmse_in_range() -> Result<()> {
    let cache = PqoCache::new(make_config(3, usize::MAX))?;
    let (k, v) = roundtrip_nmse(&cache, &cuda_device())?;
    assert_in_range("PQO3 GPU", k, v, &PQO3_TQ4_NMSE);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn pqo4_gpu_roundtrip_nmse_in_range() -> Result<()> {
    let cache = PqoCache::new(make_config(4, usize::MAX))?;
    let (k, v) = roundtrip_nmse(&cache, &cuda_device())?;
    assert_in_range("PQO4 GPU", k, v, &PQO4_NMSE);
    Ok(())
}
