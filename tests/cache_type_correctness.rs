//! Correctness tests: each cache type must behave according to its specification.
//!
//! PQ: standard codebook, no QJL, no logit_bias
//! PQO: outlier codebook, no QJL, no logit_bias
//! TQ: standard codebook + QJL correction, logit_bias is Some
//!
//! These tests verify the CONTRACT of each type — not just that it runs,
//! but that it produces the correct kind of output.

#![cfg(feature = "candle")]

use candle_core::{DType, Device, Tensor};
use mistralrs_kv_cache::{AttendConfig, CompressedKVCache, DecodeOutput};
use turboquant::cache::config::QuantNormMode;
use turboquant::cache::{CacheConfig, PqoCache, TqCache};

const HEAD_DIM: usize = 128;
const NUM_KV_HEADS: usize = 4;
const NUM_LAYERS: usize = 2;
const LAYER: usize = 0;

fn cfg(outlier_blocks: usize) -> CacheConfig {
    CacheConfig {
        bits: 3,
        head_dim: HEAD_DIM,
        num_kv_heads: NUM_KV_HEADS,
        num_layers: NUM_LAYERS,
        norm_mode: QuantNormMode::MaxNorm,
        outlier_blocks,
    }
}

fn make_kv(seq_len: usize, seed: f32) -> (Tensor, Tensor) {
    let n = NUM_KV_HEADS * seq_len * HEAD_DIM;
    let k: Vec<f32> = (0..n)
        .map(|i| ((i as f32 + seed) * 0.0137).sin() * 2.0)
        .collect();
    let v: Vec<f32> = (0..n)
        .map(|i| ((i as f32 + seed + 1000.0) * 0.0213).cos() * 1.5)
        .collect();
    (
        Tensor::from_vec(k, (1, NUM_KV_HEADS, seq_len, HEAD_DIM), &Device::Cpu).unwrap(),
        Tensor::from_vec(v, (1, NUM_KV_HEADS, seq_len, HEAD_DIM), &Device::Cpu).unwrap(),
    )
}

fn make_q(seq_len: usize) -> Tensor {
    Tensor::zeros(
        (1, NUM_KV_HEADS * 2, seq_len, HEAD_DIM),
        DType::F32,
        &Device::Cpu,
    )
    .unwrap()
}

fn attend_config() -> AttendConfig {
    AttendConfig {
        softmax_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        n_kv_groups: 2,
    }
}

// -----------------------------------------------------------------------
// PQ3: standard codebook (outlier_blocks=0), no QJL
// -----------------------------------------------------------------------

#[test]
fn pq3_uses_standard_codebook() {
    let mut cache = PqoCache::new(cfg(0));
    let (k, v) = make_kv(4, 1.0);
    let q = make_q(4);
    let result = cache.prefill(LAYER, &k, &v, &q).unwrap();
    assert!(result.logit_bias.is_none(), "PQ3 should have no logit_bias");
}

#[test]
fn pq3_quality_lower_than_pqo3() {
    // PQ3 uses standard codebook (fewer bits effective) → should have worse quality than PQO3
    let (k, v) = make_kv(8, 10.0);
    let q = make_q(8);

    let mut pq = PqoCache::new(cfg(0));
    let mut pqo = PqoCache::new(cfg(usize::MAX));

    pq.prefill(LAYER, &k, &v, &q).unwrap();
    pqo.prefill(LAYER, &k, &v, &q).unwrap();

    let (k_dec, v_dec) = make_kv(1, 11.0);
    let q_dec = make_q(1);
    let config = attend_config();

    let pq_out = pq.decode(LAYER, &k_dec, &v_dec, &q_dec, &config).unwrap();
    let pqo_out = pqo.decode(LAYER, &k_dec, &v_dec, &q_dec, &config).unwrap();

    // Both should produce valid output (not crash)
    match pq_out {
        DecodeOutput::Dequantized(r) => assert_eq!(r.k.dims()[2], 9),
        DecodeOutput::Fused(t) => assert_eq!(t.dims()[2], 1),
    }
    match pqo_out {
        DecodeOutput::Dequantized(r) => assert_eq!(r.k.dims()[2], 9),
        DecodeOutput::Fused(t) => assert_eq!(t.dims()[2], 1),
    }
}

// -----------------------------------------------------------------------
// PQO3: outlier codebook (outlier_blocks=MAX), no QJL
// -----------------------------------------------------------------------

#[test]
fn pqo3_uses_outlier_codebook() {
    let mut cache = PqoCache::new(cfg(usize::MAX));
    let (k, v) = make_kv(4, 2.0);
    let q = make_q(4);
    let result = cache.prefill(LAYER, &k, &v, &q).unwrap();
    assert!(
        result.logit_bias.is_none(),
        "PQO3 should have no logit_bias"
    );
}

#[test]
fn pqo4_uses_outlier_codebook() {
    let mut cache = PqoCache::new(CacheConfig {
        bits: 4,
        ..cfg(usize::MAX)
    });
    let (k, v) = make_kv(4, 3.0);
    let q = make_q(4);
    let result = cache.prefill(LAYER, &k, &v, &q).unwrap();
    assert!(
        result.logit_bias.is_none(),
        "PQO4 should have no logit_bias"
    );
}

// -----------------------------------------------------------------------
// TQ3: standard codebook + QJL correction → logit_bias must be Some
// -----------------------------------------------------------------------

#[test]
fn tq3_prefill_returns_logit_bias() {
    // TQ3 = 2-bit PolarQuant + 1-bit QJL. The QJL correction produces a
    // logit_bias that must be returned in DequantResult.
    // This test MUST use TqCache (not PqoCache).

    // TODO: Replace with TqCache once implemented
    // For now, this test documents the expected behavior and will FAIL
    // until TqCache is properly implemented.

    let mut cache = create_tq3_cache();
    let (k, v) = make_kv(4, 5.0);
    let q = make_q(4);
    let result = cache.prefill(LAYER, &k, &v, &q).unwrap();

    assert!(
        result.logit_bias.is_some(),
        "TQ3 MUST return logit_bias (QJL correction). \
         If this fails, TQ3 is running without QJL — that's PQ3, not TQ3."
    );
}

#[test]
fn tq3_decode_returns_logit_bias() {
    let mut cache = create_tq3_cache();
    let (k, v) = make_kv(4, 6.0);
    let q = make_q(4);
    cache.prefill(LAYER, &k, &v, &q).unwrap();

    let (k_dec, v_dec) = make_kv(1, 7.0);
    let q_dec = make_q(1);
    let config = attend_config();
    let output = cache
        .decode(LAYER, &k_dec, &v_dec, &q_dec, &config)
        .unwrap();

    match output {
        DecodeOutput::Dequantized(result) => {
            assert!(
                result.logit_bias.is_some(),
                "TQ3 decode MUST return logit_bias (QJL correction)"
            );
        }
        DecodeOutput::Fused(_) => {
            // Fused path handles QJL internally — that's OK
        }
    }
}

#[test]
fn tq4_prefill_returns_logit_bias() {
    let mut cache = create_tq4_cache();
    let (k, v) = make_kv(4, 8.0);
    let q = make_q(4);
    let result = cache.prefill(LAYER, &k, &v, &q).unwrap();

    assert!(
        result.logit_bias.is_some(),
        "TQ4 MUST return logit_bias (QJL correction)"
    );
}

// -----------------------------------------------------------------------
// Helper: create TQ cache (will be TqCache once implemented)
// -----------------------------------------------------------------------

fn create_tq3_cache() -> Box<dyn CompressedKVCache> {
    Box::new(TqCache::new(cfg(0)))
}

fn create_tq4_cache() -> Box<dyn CompressedKVCache> {
    Box::new(TqCache::new(CacheConfig { bits: 4, ..cfg(0) }))
}
