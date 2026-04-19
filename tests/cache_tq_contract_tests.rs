//! TqCache contract tests: QJL correction must produce a logit_bias.
//!
//! Extracted from the former `cache_type_correctness.rs`.

#![cfg(feature = "candle")]

use candle_core::{DType, Device, Tensor};
use mistralrs_kv_cache::{AttendConfig, CompressedKVCache, DecodeOutput};
use turboquant::cache::config::QuantNormMode;
use turboquant::cache::{CacheConfig, TqCache};
use turboquant::test_utils::make_kv as shared_make_kv;

const HEAD_DIM: usize = 128;
const NUM_KV_HEADS: usize = 4;
const NUM_LAYERS: usize = 2;
const LAYER: usize = 0;

fn cfg(bits: u8) -> CacheConfig {
    CacheConfig {
        bits,
        head_dim: HEAD_DIM,
        num_kv_heads: NUM_KV_HEADS,
        num_layers: NUM_LAYERS,
        norm_mode: QuantNormMode::MaxNorm,
        outlier_blocks: 0,
    }
}

fn make_kv(seq_len: usize, seed: u32) -> (Tensor, Tensor) {
    shared_make_kv(seq_len, NUM_KV_HEADS, HEAD_DIM, seed)
}

fn make_q(seq_len: usize) -> Tensor {
    Tensor::zeros(
        (1, NUM_KV_HEADS * 2, seq_len, HEAD_DIM),
        DType::F32,
        &Device::Cpu,
    )
    .unwrap()
}

fn create_tq_cache(bits: u8) -> Box<dyn CompressedKVCache> {
    Box::new(TqCache::new(cfg(bits)).unwrap())
}

#[test]
fn tq_prefill_returns_logit_bias_for_every_bit_width() {
    // TQ3 / TQ4: QJL correction must produce a logit_bias regardless of bit width.
    // If this fails with logit_bias=None, the cache is running without QJL
    // (i.e. it's acting like PQ, not TQ).
    for (bits, seed) in [(3u8, 5u32), (4u8, 8u32)] {
        let cache = create_tq_cache(bits);
        let (k, v) = make_kv(4, seed);
        let q = make_q(4);
        let result = cache.prefill(LAYER, &k, &v, &q).unwrap();
        assert!(
            result.logit_bias.is_some(),
            "TQ{bits} MUST return logit_bias (QJL correction)"
        );
    }
}

#[test]
fn tq3_decode_returns_logit_bias() {
    let cache = create_tq_cache(3);
    let (k, v) = make_kv(4, 6);
    let q = make_q(4);
    cache.prefill(LAYER, &k, &v, &q).unwrap();

    let (k_dec, v_dec) = make_kv(1, 7);
    let q_dec = make_q(1);
    let config = AttendConfig {
        softmax_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        n_kv_groups: 2,
    };
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
            // Fused path handles QJL internally — that's OK.
        }
    }
}
