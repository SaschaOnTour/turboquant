//! PqoCache state-isolation tests — reset and per-layer independence.
//!
//! Extracted from the former `cache_pqo_tests.rs`.

#![cfg(feature = "candle")]

use candle_core::Tensor;
use mistralrs_kv_cache::CompressedKVCache;
use turboquant::cache::config::QuantNormMode;
use turboquant::cache::{CacheConfig, PqoCache};
use turboquant::test_utils::{make_kv as shared_make_kv, make_q as shared_make_q};

const HEAD_DIM: usize = 128;
const NUM_KV_HEADS: usize = 8;
const NUM_ATTN_HEADS: usize = NUM_KV_HEADS * 2;
const NUM_LAYERS: usize = 2;
const BITS: u8 = 3;

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
fn pqo3_reset_clears_all_layers() -> candle_core::Result<()> {
    let cache = PqoCache::new(pqo_config())?;
    let (k, v) = make_kv(4, 11);
    let q = make_q(4);

    cache.prefill(0, &k, &v, &q).unwrap();
    cache.prefill(1, &k, &v, &q).unwrap();
    assert_eq!(cache.seq_len(0), 4);
    assert_eq!(cache.seq_len(1), 4);

    cache.reset().unwrap();
    assert_eq!(cache.seq_len(0), 0);
    assert_eq!(cache.seq_len(1), 0);
    assert_eq!(cache.memory_usage(), 0);
    Ok(())
}

#[test]
fn pqo3_layers_are_independent() -> candle_core::Result<()> {
    let cache = PqoCache::new(pqo_config())?;
    let (k4, v4) = make_kv(4, 12);
    let (k8, v8) = make_kv(8, 13);
    let q4 = make_q(4);
    let q8 = make_q(8);

    cache.prefill(0, &k4, &v4, &q4).unwrap();
    cache.prefill(1, &k8, &v8, &q8).unwrap();

    assert_eq!(cache.seq_len(0), 4);
    assert_eq!(cache.seq_len(1), 8);
    Ok(())
}
