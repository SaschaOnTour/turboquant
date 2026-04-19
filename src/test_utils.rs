//! Shared test utilities.
//!
//! This module is `#[doc(hidden)] pub` so that integration tests, benches,
//! and examples in the same crate can import helpers without each
//! redefining them. It is NOT part of the public API.

#[cfg(feature = "candle")]
use candle_core::{Device, Tensor};

/// LCG multiplier for pseudo-random vector generation.
pub const LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;
/// LCG increment.
pub const LCG_INCREMENT: u64 = 1;
/// Right-shift for extracting bits from LCG state.
pub const LCG_SHIFT: u32 = 33;

/// Deterministic pseudo-random vector for tests. Uses LCG, no rand dependency.
pub fn pseudo_random_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..dim)
        .map(|_| {
            state = state
                .wrapping_mul(LCG_MULTIPLIER)
                .wrapping_add(LCG_INCREMENT);
            let bits = (state >> LCG_SHIFT) as i32;
            bits as f32 / (i32::MAX as f32)
        })
        .collect()
}

/// Frequency factor for the sine-based K data generator in `make_kv`.
pub const K_SEED_FREQ: f32 = 0.0137;
/// Frequency factor for the cosine-based V data generator in `make_kv`.
pub const V_SEED_FREQ: f32 = 0.0213;
/// Seed offset to decorrelate V from K in `make_kv`.
pub const V_SEED_OFFSET: f32 = 1000.0;
/// Peak amplitude of the generated K values in `make_kv`.
pub const K_AMPLITUDE: f32 = 2.0;
/// Peak amplitude of the generated V values in `make_kv`.
pub const V_AMPLITUDE: f32 = 1.5;

/// Generate deterministic `(K, V)` test tensors of shape
/// `[1, num_kv_heads, seq_len, head_dim]`, using a sine/cosine generator
/// seeded by `seed` so each test can produce distinct-but-reproducible data.
#[cfg(feature = "candle")]
pub fn make_kv(
    seq_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seed: u32,
) -> (Tensor, Tensor) {
    let n = num_kv_heads * seq_len * head_dim;
    let s = seed as f32;
    let k_data: Vec<f32> = (0..n)
        .map(|i| ((i as f32 + s) * K_SEED_FREQ).sin() * K_AMPLITUDE)
        .collect();
    let v_data: Vec<f32> = (0..n)
        .map(|i| ((i as f32 + s + V_SEED_OFFSET) * V_SEED_FREQ).cos() * V_AMPLITUDE)
        .collect();
    let k = Tensor::from_vec(k_data, (1, num_kv_heads, seq_len, head_dim), &Device::Cpu).unwrap();
    let v = Tensor::from_vec(v_data, (1, num_kv_heads, seq_len, head_dim), &Device::Cpu).unwrap();
    (k, v)
}

/// Zero-filled query tensor `[1, num_attn_heads, seq_len, head_dim]` — used
/// by cache tests that don't exercise the Q path (e.g. PQO without QJL).
#[cfg(feature = "candle")]
pub fn make_q(seq_len: usize, num_attn_heads: usize, head_dim: usize) -> Tensor {
    Tensor::zeros(
        (1, num_attn_heads, seq_len, head_dim),
        candle_core::DType::F32,
        &Device::Cpu,
    )
    .unwrap()
}

/// Cosine similarity between two tensors, flattened to f32.
#[cfg(feature = "candle")]
pub fn cosine_sim(a: &Tensor, b: &Tensor) -> f32 {
    let a_flat: Vec<f32> = a
        .to_dtype(candle_core::DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let b_flat: Vec<f32> = b
        .to_dtype(candle_core::DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let dot: f32 = a_flat.iter().zip(b_flat.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a_flat.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b_flat.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

// ---------------------------------------------------------------------------
// SplitMix64: high-quality 64-bit generator used for paper-verification
// statistical tests. Same finalizer as turboquant-rs's Rademacher signs.
// ---------------------------------------------------------------------------

/// SplitMix64 gamma constant (Stafford variant 13).
pub const SPLITMIX_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;
/// SplitMix64 mix multiplier 1.
pub const SPLITMIX_MUL1: u64 = 0xbf58_476d_1ce4_e5b9;
/// SplitMix64 mix multiplier 2.
pub const SPLITMIX_MUL2: u64 = 0x94d0_49bb_1331_11eb;
/// First xor-shift amount for the SplitMix64 finalizer.
pub const SPLITMIX_SHIFT_1: u32 = 30;
/// Second xor-shift amount for the SplitMix64 finalizer.
pub const SPLITMIX_SHIFT_2: u32 = 27;
/// Third xor-shift amount for the SplitMix64 finalizer.
pub const SPLITMIX_SHIFT_3: u32 = 31;
/// Number of mantissa bits in an f64 (IEEE-754).
pub const F64_MANTISSA_BITS: u32 = 53;
/// Right-shift to keep exactly `F64_MANTISSA_BITS` bits of a u64.
pub const U64_TO_F64_SHIFT: u32 = 64 - F64_MANTISSA_BITS;

/// Deterministic 64-bit PRNG. Used by paper-verification tests to drive
/// Box-Muller Gaussian sampling with full 64-bit entropy.
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(SPLITMIX_GAMMA);
        let mut z = self.state;
        z = (z ^ (z >> SPLITMIX_SHIFT_1)).wrapping_mul(SPLITMIX_MUL1);
        z = (z ^ (z >> SPLITMIX_SHIFT_2)).wrapping_mul(SPLITMIX_MUL2);
        z ^ (z >> SPLITMIX_SHIFT_3)
    }

    /// Returns an f64 in (0, 1), never exactly 0 or 1.
    pub fn next_open01(&mut self) -> f64 {
        ((self.next_u64() >> U64_TO_F64_SHIFT) as f64 + 0.5) / (1u64 << F64_MANTISSA_BITS) as f64
    }
}

/// Deterministic unit vector on S^{d-1} via Box-Muller (Gaussian coordinates,
/// then normalise). Used by paper-theorem statistical tests.
pub fn random_unit_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    let mut gaussians = Vec::with_capacity(dim);

    let pairs = dim.div_ceil(2);
    for _ in 0..pairs {
        let u1 = rng.next_open01();
        let u2 = rng.next_open01();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        gaussians.push(r * theta.cos());
        gaussians.push(r * theta.sin());
    }
    gaussians.truncate(dim);

    let norm: f64 = gaussians.iter().map(|x| x * x).sum::<f64>().sqrt();
    gaussians.iter().map(|x| (*x / norm) as f32).collect()
}

/// Unnormalised pseudo-random vector using SplitMix64 (full-range f32).
/// Used by the paper's WHT tests.
#[allow(dead_code)] // paper-WHT-only helper; rustqual Bug 1 false-positive
pub fn splitmix_random_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    (0..dim)
        .map(|_| (rng.next_u64() as i64) as f32 / (i64::MAX as f32))
        .collect()
}

/// i.i.d. standard normal samples of length `dim`, produced by SplitMix64 +
/// Box-Muller. Used by MSE validation tests that need raw Gaussian inputs.
pub fn random_normal_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    let mut out = Vec::with_capacity(dim);
    let pairs = dim.div_ceil(2);
    for _ in 0..pairs {
        let u1 = rng.next_open01();
        let u2 = rng.next_open01();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        out.push((r * theta.cos()) as f32);
        if out.len() < dim {
            out.push((r * theta.sin()) as f32);
        }
    }
    out
}
