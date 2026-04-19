//! PqoCache prefill tests — return semantics & seq_len tracking.
//!
//! Extracted from the former `cache_pqo_tests.rs`.

#![cfg(feature = "candle")]

use candle_core::Tensor;
use mistralrs_kv_cache::CompressedKVCache;
use turboquant::cache::config::QuantNormMode;
use turboquant::cache::{CacheConfig, PqoCache};
use turboquant::test_utils::{cosine_sim, make_kv as shared_make_kv, make_q as shared_make_q};

const HEAD_DIM: usize = 128;
const NUM_KV_HEADS: usize = 8;
const NUM_ATTN_HEADS: usize = NUM_KV_HEADS * 2;
const NUM_LAYERS: usize = 2;
const BITS: u8 = 3;
const TEST_LAYER: usize = 0;

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
fn pqo3_prefill_returns_original_on_first_call() -> candle_core::Result<()> {
    let cache = PqoCache::new(pqo_config())?;
    let (k, v) = make_kv(8, 1);
    let q = make_q(8);

    let result = cache.prefill(TEST_LAYER, &k, &v, &q).unwrap();

    assert_eq!(result.k.dims(), k.dims());
    assert_eq!(result.v.dims(), v.dims());
    let sim = cosine_sim(&result.k, &k);
    assert!(
        sim > 0.999,
        "First prefill should return originals, got cosine_sim={sim}"
    );
    assert!(result.logit_bias.is_none(), "PQO should have no logit_bias");
    Ok(())
}

#[test]
fn pqo3_prefill_updates_seq_len() -> candle_core::Result<()> {
    let cache = PqoCache::new(pqo_config())?;
    let (k, v) = make_kv(16, 2);
    let q = make_q(16);

    assert_eq!(cache.seq_len(TEST_LAYER), 0);
    cache.prefill(TEST_LAYER, &k, &v, &q).unwrap();
    assert_eq!(cache.seq_len(TEST_LAYER), 16);
    Ok(())
}
