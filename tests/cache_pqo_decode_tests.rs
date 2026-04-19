//! PqoCache decode tests — CPU decode returns dequantized KV for SDPA.
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
fn pqo3_decode_returns_dequantized() -> candle_core::Result<()> {
    let cache = PqoCache::new(pqo_config())?;

    // Prefill 8 tokens.
    let (k_pre, v_pre) = make_kv(8, 3);
    let q_pre = make_q(8);
    cache.prefill(TEST_LAYER, &k_pre, &v_pre, &q_pre).unwrap();

    // Decode 1 token.
    let (k_dec, v_dec) = make_kv(1, 4);
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
            // Should contain all 9 tokens (8 prefill + 1 decode).
            assert_eq!(result.k.dims(), &[1, NUM_KV_HEADS, 9, HEAD_DIM]);
            assert_eq!(result.v.dims(), &[1, NUM_KV_HEADS, 9, HEAD_DIM]);
            assert!(result.logit_bias.is_none());
        }
        DecodeOutput::Fused(_) => panic!("CPU should not use fused path"),
    }
    assert_eq!(cache.seq_len(TEST_LAYER), 9);
    Ok(())
}
