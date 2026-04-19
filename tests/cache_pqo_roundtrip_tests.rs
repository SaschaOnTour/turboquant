//! PqoCache roundtrip quality tests — parametric by (bits, norm-mode, threshold).
//!
//! Extracted from the former `cache_pqo_tests.rs`.

#![cfg(feature = "candle")]

use candle_core::Tensor;
use mistralrs_kv_cache::{AttendConfig, CompressedKVCache, DecodeOutput};
use turboquant::cache::config::QuantNormMode;
use turboquant::cache::{CacheConfig, PqoCache};
use turboquant::test_utils::{cosine_sim, make_kv as shared_make_kv, make_q as shared_make_q};

const HEAD_DIM: usize = 128;
const NUM_KV_HEADS: usize = 8;
const NUM_ATTN_HEADS: usize = NUM_KV_HEADS * 2;
const NUM_LAYERS: usize = 2;
const TEST_LAYER: usize = 0;
const N_KV_GROUPS: usize = 2;
const PREFILL_LEN: usize = 4;

/// Minimum cosine similarity for PQO3 with MaxNorm scaling. Derived empirically
/// from the codebook's lossless-region and block-wise scale granularity.
const PQO3_MAXNORM_MIN_SIM: f32 = 0.85;
/// Minimum cosine similarity for PQO3 with L2-norm scaling (slightly looser
/// because L2 scale is less representative for sparse high-norm blocks).
const PQO3_L2NORM_MIN_SIM: f32 = 0.83;
/// Minimum cosine similarity for PQO4 with MaxNorm scaling (4-bit codebook
/// is ~16× finer than 3-bit, so quality is substantially higher).
const PQO4_MAXNORM_MIN_SIM: f32 = 0.92;

fn pqo_config(bits: u8, norm_mode: QuantNormMode) -> CacheConfig {
    CacheConfig {
        bits,
        head_dim: HEAD_DIM,
        num_kv_heads: NUM_KV_HEADS,
        num_layers: NUM_LAYERS,
        norm_mode,
        outlier_blocks: usize::MAX,
    }
}

fn make_kv(seq_len: usize, seed: u32) -> (Tensor, Tensor) {
    shared_make_kv(seq_len, NUM_KV_HEADS, HEAD_DIM, seed)
}

fn make_q(seq_len: usize) -> Tensor {
    shared_make_q(seq_len, NUM_ATTN_HEADS, HEAD_DIM)
}

/// Run a prefill-then-decode quality check: prefill `PREFILL_LEN` tokens,
/// decode 1 token, and verify the dequantized K has cosine similarity
/// above `min_sim` against the ground-truth concatenation.
fn roundtrip_quality_check(
    bits: u8,
    norm_mode: QuantNormMode,
    min_sim: f32,
    seed_pre: u32,
    seed_dec: u32,
) {
    let cache = PqoCache::new(pqo_config(bits, norm_mode)).expect("cache::new");
    let (k_pre, v_pre) = make_kv(PREFILL_LEN, seed_pre);
    let q = make_q(PREFILL_LEN);
    cache.prefill(TEST_LAYER, &k_pre, &v_pre, &q).unwrap();

    let (k_dec, v_dec) = make_kv(1, seed_dec);
    let q_dec = make_q(1);
    let config = AttendConfig {
        softmax_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        n_kv_groups: N_KV_GROUPS,
    };
    let output = cache
        .decode(TEST_LAYER, &k_dec, &v_dec, &q_dec, &config)
        .unwrap();

    match output {
        DecodeOutput::Dequantized(result) => {
            let k_orig = Tensor::cat(&[&k_pre, &k_dec], 2).unwrap();
            let sim = cosine_sim(&result.k, &k_orig);
            assert!(
                sim > min_sim,
                "PQO{bits} {norm_mode:?} roundtrip cosine_sim={sim:.4}, expected > {min_sim}"
            );
        }
        DecodeOutput::Fused(_) => panic!("Expected Dequantized on CPU, got Fused"),
    }
}

#[test]
fn pqo3_roundtrip_quality_maxnorm() {
    roundtrip_quality_check(3, QuantNormMode::MaxNorm, PQO3_MAXNORM_MIN_SIM, 5, 6);
}

#[test]
fn pqo3_roundtrip_quality_l2norm() {
    roundtrip_quality_check(3, QuantNormMode::L2Norm, PQO3_L2NORM_MIN_SIM, 7, 8);
}

#[test]
fn pqo4_roundtrip_quality_maxnorm() {
    roundtrip_quality_check(4, QuantNormMode::MaxNorm, PQO4_MAXNORM_MIN_SIM, 9, 10);
}
