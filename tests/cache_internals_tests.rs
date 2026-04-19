//! Integration tests for internal cache helpers.
//!
//! Helpers are marked `#[doc(hidden)] pub` — Rust convention for items
//! that are reachable from integration tests but not part of the public
//! API (no SemVer guarantees).

#![cfg(feature = "candle")]

// qual:allow(srp) — cohesive integration-test module
use candle_core::Device;
use turboquant::cache::config::QuantNormMode;
use turboquant::cache::{CacheConfig, GpuPrecomputed, PrecomputedState};

fn test_config() -> CacheConfig {
    CacheConfig {
        bits: 3,
        head_dim: 128,
        num_kv_heads: 4,
        num_layers: 2,
        norm_mode: QuantNormMode::MaxNorm,
        outlier_blocks: usize::MAX,
    }
}

#[test]
fn ensure_gpu_precomputed() {
    let state = PrecomputedState::default();
    let cfg = test_config();
    let device = Device::Cpu;

    // First call initializes.
    let p1 = turboquant::cache::ensure_gpu_precomputed(&state, &cfg, &device).unwrap();
    let p1_addr = p1 as *const GpuPrecomputed;
    // Second call returns the same instance (no re-init).
    let p2 = turboquant::cache::ensure_gpu_precomputed(&state, &cfg, &device).unwrap();
    let p2_addr = p2 as *const GpuPrecomputed;
    assert_eq!(
        p1_addr, p2_addr,
        "concurrent init returned a different instance"
    );
}

#[test]
fn ensure_gpu_precomputed_returns_initialized_cell() {
    let state = PrecomputedState::default();
    let cfg = test_config();
    let device = Device::Cpu;

    let p = turboquant::cache::ensure_gpu_precomputed(&state, &cfg, &device).unwrap();
    // Precomputed should carry metadata matching config.
    assert!(p.outlier_centroids.dims()[0] > 0);
}
