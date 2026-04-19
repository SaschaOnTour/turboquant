//! `PqoCache` storage-integration tests — verifies that prefill drives
//! `dequantize_full_impl` once storage holds data.
//!
//! Extracted from the former `cache_storage_tests.rs` (per-layer and
//! metadata tests moved to `layer_storage_tests.rs` and
//! `storage_metadata_tests.rs`).

#![cfg(feature = "candle")]

use candle_core::{DType, Device, Tensor};
use mistralrs_kv_cache::CompressedKVCache;
use turboquant::cache::config::QuantNormMode;
use turboquant::cache::{CacheConfig, PqoCache};

const HEAD_DIM: usize = 128;
const NUM_KV_HEADS: usize = 4;
const BITS: u8 = 3;

fn make_kv(seq_len: usize) -> (Tensor, Tensor) {
    let n = NUM_KV_HEADS * seq_len * HEAD_DIM;
    let k: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
    let v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.02).cos()).collect();
    let k = Tensor::from_vec(k, (1, NUM_KV_HEADS, seq_len, HEAD_DIM), &Device::Cpu).unwrap();
    let v = Tensor::from_vec(v, (1, NUM_KV_HEADS, seq_len, HEAD_DIM), &Device::Cpu).unwrap();
    (k, v)
}

fn make_q(seq_len: usize) -> Tensor {
    Tensor::zeros(
        (1, NUM_KV_HEADS, seq_len, HEAD_DIM),
        DType::F32,
        &Device::Cpu,
    )
    .unwrap()
}

#[test]
fn dequantize_full_roundtrip_produces_output() -> candle_core::Result<()> {
    let cache = PqoCache::new(CacheConfig {
        bits: BITS,
        head_dim: HEAD_DIM,
        num_kv_heads: NUM_KV_HEADS,
        num_layers: 2,
        norm_mode: QuantNormMode::MaxNorm,
        outlier_blocks: usize::MAX,
    })?;
    let (k, v) = make_kv(8);
    let q = make_q(8);

    // First prefill returns originals (empty cache).
    let result = cache.prefill(0, &k, &v, &q).unwrap();
    assert_eq!(result.k.dims(), k.dims());

    // Second prefill drives `dequantize_full_impl`.
    let (k2, v2) = make_kv(4);
    let q2 = make_q(4);
    let result2 = cache.prefill(0, &k2, &v2, &q2).unwrap();
    assert_eq!(result2.k.dims()[2], 12); // 8 + 4
    assert!(result2.logit_bias.is_none());
    Ok(())
}
