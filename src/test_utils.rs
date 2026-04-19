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
#[allow(dead_code)] // rustqual flags as testonly; real callers in tests/+benches/+examples (see docs/rustqual-bugs.md)
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
#[allow(dead_code)] // used across tests/+benches/ (see docs/rustqual-bugs.md)
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
