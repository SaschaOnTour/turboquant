//! PqoCache contract tests — standard (non-outlier) codebook path.
//!
//! `PqoCache` with `outlier_blocks = 0` is the "PQ" variant.
//! Extracted from the former `cache_pqo_contract_tests.rs`.

#![cfg(feature = "candle")]

use candle_core::{DType, Device, Tensor};
use mistralrs_kv_cache::{AttendConfig, CompressedKVCache, DecodeOutput};
use turboquant::cache::config::QuantNormMode;
use turboquant::cache::{CacheConfig, PqoCache};
use turboquant::test_utils::make_kv as shared_make_kv;

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

fn attend_config() -> AttendConfig {
    AttendConfig {
        softmax_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        n_kv_groups: 2,
    }
}

#[test]
fn pq3_uses_standard_codebook() -> candle_core::Result<()> {
    let cache = PqoCache::new(cfg(0))?;
    let (k, v) = make_kv(4, 1);
    let q = make_q(4);
    let result = cache.prefill(LAYER, &k, &v, &q).unwrap();
    assert!(result.logit_bias.is_none(), "PQ3 should have no logit_bias");
    Ok(())
}

#[test]
fn pq3_and_pqo3_both_produce_valid_output() -> candle_core::Result<()> {
    let (k, v) = make_kv(8, 10);
    let q = make_q(8);

    let pq = PqoCache::new(cfg(0))?;
    let pqo = PqoCache::new(cfg(usize::MAX))?;

    pq.prefill(LAYER, &k, &v, &q).unwrap();
    pqo.prefill(LAYER, &k, &v, &q).unwrap();

    let (k_dec, v_dec) = make_kv(1, 11);
    let q_dec = make_q(1);
    let config = attend_config();

    let pq_out = pq.decode(LAYER, &k_dec, &v_dec, &q_dec, &config).unwrap();
    let pqo_out = pqo.decode(LAYER, &k_dec, &v_dec, &q_dec, &config).unwrap();

    match pq_out {
        DecodeOutput::Dequantized(r) => assert_eq!(r.k.dims()[2], 9),
        DecodeOutput::Fused(t) => assert_eq!(t.dims()[2], 1),
    }
    match pqo_out {
        DecodeOutput::Dequantized(r) => assert_eq!(r.k.dims()[2], 9),
        DecodeOutput::Fused(t) => assert_eq!(t.dims()[2], 1),
    }
    Ok(())
}
