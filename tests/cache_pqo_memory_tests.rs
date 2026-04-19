//! PqoCache memory-usage tests and multi-step decode sanity.
//!
//! Extracted from the former `cache_pqo_tests.rs`.

#![cfg(feature = "candle")]

use candle_core::Tensor;
use mistralrs_kv_cache::{AttendConfig, CompressedKVCache, DecodeOutput};
use turboquant::cache::config::QuantNormMode;
use turboquant::cache::{CacheConfig, PqoCache};
use turboquant::test_utils::{make_kv as shared_make_kv, make_q as shared_make_q};

const HEAD_DIM: usize = 128;
const NUM_KV_HEADS: usize = 8;
const NUM_ATTN_HEADS: usize = NUM_KV_HEADS * 2;
const NUM_LAYERS: usize = 2;
const BITS: u8 = 3;
const TEST_LAYER: usize = 0;
const N_KV_GROUPS: usize = 2;

/// Packed dim per token for PQO3 at HEAD_DIM=128: 3 bits × 128 / 8 = 48 bytes.
const PQO3_PACKED_BYTES_PER_TOKEN: usize = 48;

fn pqo_config() -> CacheConfig {
    CacheConfig {
        bits: BITS,
        head_dim: HEAD_DIM,
        num_kv_heads: NUM_KV_HEADS,
        num_layers: NUM_LAYERS,
        norm_mode: QuantNormMode::MaxNorm,
        outlier_blocks: usize::MAX,
    }
}

fn make_kv(seq_len: usize, seed: u32) -> (Tensor, Tensor) {
    shared_make_kv(seq_len, NUM_KV_HEADS, HEAD_DIM, seed)
}

fn make_q(seq_len: usize) -> Tensor {
    shared_make_q(seq_len, NUM_ATTN_HEADS, HEAD_DIM)
}

#[test]
fn pqo3_memory_usage_increases_with_tokens() -> candle_core::Result<()> {
    let cache = PqoCache::new(pqo_config())?;
    assert_eq!(cache.memory_usage(), 0);

    let seq = 16;
    let (k, v) = make_kv(seq, 14);
    let q = make_q(seq);
    cache.prefill(TEST_LAYER, &k, &v, &q).unwrap();

    let usage = cache.memory_usage();
    assert!(usage > 0, "Memory usage should be > 0 after prefill");

    // PQO3: 3-bit packed indices alone should exceed this lower bound.
    let expected_min = NUM_KV_HEADS * seq * PQO3_PACKED_BYTES_PER_TOKEN;
    assert!(
        usage > expected_min,
        "Memory usage {usage} too low, expected > {expected_min}"
    );
    Ok(())
}

#[test]
fn pqo3_multi_step_decode() -> candle_core::Result<()> {
    let cache = PqoCache::new(pqo_config())?;
    let config = AttendConfig {
        softmax_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        n_kv_groups: N_KV_GROUPS,
    };

    // Prefill 4 tokens, then decode 10 more one by one.
    let (k_pre, v_pre) = make_kv(4, 15);
    let q_pre = make_q(4);
    cache.prefill(TEST_LAYER, &k_pre, &v_pre, &q_pre).unwrap();

    for step in 0..10 {
        let (k_dec, v_dec) = make_kv(1, 16 + step as u32);
        let q_dec = make_q(1);
        let output = cache
            .decode(TEST_LAYER, &k_dec, &v_dec, &q_dec, &config)
            .unwrap();

        match output {
            DecodeOutput::Dequantized(result) => {
                let expected_seq = 4 + step + 1;
                assert_eq!(
                    result.k.dims(),
                    &[1, NUM_KV_HEADS, expected_seq, HEAD_DIM],
                    "Step {step}: wrong K shape"
                );
            }
            DecodeOutput::Fused(_) => panic!("CPU should not use fused path"),
        }
    }
    assert_eq!(cache.seq_len(TEST_LAYER), 14);
    Ok(())
}
