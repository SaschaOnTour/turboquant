//! PqoCache contract tests — outlier codebook path.
//!
//! `PqoCache` with `outlier_blocks = usize::MAX` is the "PQO" variant.
//! Extracted from the former `cache_pqo_contract_tests.rs`.

#![cfg(feature = "candle")]

use candle_core::{DType, Device, Tensor};
use mistralrs_kv_cache::CompressedKVCache;
use turboquant::cache::config::QuantNormMode;
use turboquant::cache::{CacheConfig, PqoCache};
use turboquant::test_utils::make_kv as shared_make_kv;

const HEAD_DIM: usize = 128;
const NUM_KV_HEADS: usize = 4;
const NUM_LAYERS: usize = 2;
const LAYER: usize = 0;

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

#[test]
fn pqo_uses_outlier_codebook_across_bit_widths() -> candle_core::Result<()> {
    // PQO3 + PQO4: outlier codebook is active (outlier_blocks = usize::MAX).
    // Neither path emits a logit_bias — that's the TQ contract.
    for (bits, seed) in [(3u8, 2u32), (4u8, 3u32)] {
        let cache = PqoCache::new(CacheConfig {
            bits,
            head_dim: HEAD_DIM,
            num_kv_heads: NUM_KV_HEADS,
            num_layers: NUM_LAYERS,
            norm_mode: QuantNormMode::MaxNorm,
            outlier_blocks: usize::MAX,
        })?;
        let (k, v) = make_kv(4, seed);
        let q = make_q(4);
        let result = cache.prefill(LAYER, &k, &v, &q).unwrap();
        assert!(
            result.logit_bias.is_none(),
            "PQO{bits} should have no logit_bias"
        );
    }
    Ok(())
}
