//! Unit tests for CompressedStorage accessors and common cache helpers.
//!
//! Covers: is_active, k_indices, k_scales, v_indices, v_scales, kv_heads,
//! reset, dequantize_full_impl, dequant_result.

#![cfg(feature = "candle")]

use candle_core::{DType, Device, Tensor};
use mistralrs_kv_cache::CompressedKVCache;
use turboquant::cache::config::QuantNormMode;
use turboquant::cache::{CacheConfig, CompressedStorage, PqoCache, QuantizedKV};

const HEAD_DIM: usize = 128;
const NUM_KV_HEADS: usize = 4;
const NUM_LAYERS: usize = 2;
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

// -- CompressedStorage tests ------------------------------------------------

#[test]
fn storage_new_is_empty() {
    let storage = CompressedStorage::new(NUM_KV_HEADS, HEAD_DIM, BITS, NUM_LAYERS);
    assert_eq!(storage.seq_len(0), 0);
    assert!(!storage.is_active(0));
    assert!(storage.k_indices(0).is_none());
    assert!(storage.k_scales(0).is_none());
    assert!(storage.v_indices(0).is_none());
    assert!(storage.v_scales(0).is_none());
}

#[test]
fn storage_after_capacity_and_append() {
    let mut storage = CompressedStorage::new(NUM_KV_HEADS, HEAD_DIM, BITS, NUM_LAYERS);
    let packed_dim = storage.packed_dim();
    let num_blocks = storage.num_blocks();
    let seq = 4;

    storage.ensure_capacity(0, seq, &Device::Cpu).unwrap();
    // After ensure_capacity but before append, indices exist but is_active is false
    assert!(!storage.is_active(0));

    let ki = Tensor::zeros((NUM_KV_HEADS, seq, packed_dim), DType::U8, &Device::Cpu).unwrap();
    let ks = Tensor::zeros((NUM_KV_HEADS, seq, num_blocks), DType::F16, &Device::Cpu).unwrap();
    let vi = Tensor::zeros((NUM_KV_HEADS, seq, packed_dim), DType::U8, &Device::Cpu).unwrap();
    let vs = Tensor::zeros((NUM_KV_HEADS, seq, num_blocks), DType::F16, &Device::Cpu).unwrap();
    let kv = QuantizedKV {
        k_indices: &ki,
        k_scales: &ks,
        v_indices: &vi,
        v_scales: &vs,
    };
    storage.append(0, 0, &kv, seq).unwrap();

    assert!(storage.is_active(0));
    assert_eq!(storage.seq_len(0), seq);
    assert!(storage.k_indices(0).is_some());
    assert!(storage.k_scales(0).is_some());
    assert!(storage.v_indices(0).is_some());
    assert!(storage.v_scales(0).is_some());
}

#[test]
fn storage_head_dim_and_bits() {
    let storage = CompressedStorage::new(NUM_KV_HEADS, HEAD_DIM, BITS, NUM_LAYERS);
    // packed_dim = head_dim * bits / 8 = 128 * 3 / 8 = 48
    assert_eq!(storage.packed_dim(), 48);
    // num_blocks = head_dim / 32 = 4
    assert_eq!(storage.num_blocks(), 4);
}

#[test]
fn storage_reset_clears_all() {
    let mut storage = CompressedStorage::new(NUM_KV_HEADS, HEAD_DIM, BITS, NUM_LAYERS);
    let packed_dim = storage.packed_dim();
    let num_blocks = storage.num_blocks();
    let seq = 2;

    storage.ensure_capacity(0, seq, &Device::Cpu).unwrap();
    let ki = Tensor::zeros((NUM_KV_HEADS, seq, packed_dim), DType::U8, &Device::Cpu).unwrap();
    let ks = Tensor::zeros((NUM_KV_HEADS, seq, num_blocks), DType::F16, &Device::Cpu).unwrap();
    let vi = ki.clone();
    let vs = ks.clone();
    let kv = QuantizedKV {
        k_indices: &ki,
        k_scales: &ks,
        v_indices: &vi,
        v_scales: &vs,
    };
    storage.append(0, 0, &kv, seq).unwrap();
    assert!(storage.is_active(0));

    storage.reset();
    assert!(!storage.is_active(0));
    assert_eq!(storage.seq_len(0), 0);
    assert!(storage.k_indices(0).is_none());
}

// -- dequantize_full_impl via PqoCache roundtrip ----------------------------

#[test]
fn dequantize_full_roundtrip_produces_output() {
    let mut cache = PqoCache::new(CacheConfig {
        bits: BITS,
        head_dim: HEAD_DIM,
        num_kv_heads: NUM_KV_HEADS,
        num_layers: NUM_LAYERS,
        norm_mode: QuantNormMode::MaxNorm,
        outlier_blocks: usize::MAX,
    })?;
    let (k, v) = make_kv(8);
    let q = make_q(8);

    // Prefill to populate storage
    let result = cache.prefill(0, &k, &v, &q).unwrap();
    // First prefill returns originals
    assert_eq!(result.k.dims(), k.dims());

    // Second prefill triggers dequantize_full_impl
    let (k2, v2) = make_kv(4);
    let q2 = make_q(4);
    let result2 = cache.prefill(0, &k2, &v2, &q2).unwrap();
    // Full dequant returns [1, heads, total_seq, dim]
    assert_eq!(result2.k.dims()[2], 12); // 8 + 4
    assert!(result2.logit_bias.is_none());
}
