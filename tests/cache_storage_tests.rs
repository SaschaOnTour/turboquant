//! Unit tests for LayerStorage accessors and common cache helpers.
//!
//! Covers per-layer storage directly (the type exposed to cache impls),
//! plus a roundtrip integration test through PqoCache.

#![cfg(feature = "candle")]

// qual:allow(srp) — cohesive integration-test module
use candle_core::{DType, Device, Tensor};
use mistralrs_kv_cache::CompressedKVCache;
use turboquant::cache::config::QuantNormMode;
use turboquant::cache::{CacheConfig, LayerStorage, PqoCache, QuantizedKV, StorageMetadata};

const HEAD_DIM: usize = 128;
const NUM_KV_HEADS: usize = 4;
const BITS: u8 = 3;

fn metadata() -> StorageMetadata {
    StorageMetadata {
        num_kv_heads: NUM_KV_HEADS,
        head_dim: HEAD_DIM,
        bits: BITS,
    }
}

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

// -- StorageMetadata tests --------------------------------------------------

#[test]
fn metadata_derives_packing_params() {
    let m = metadata();
    // packed_dim = head_dim * bits / 8 = 128 * 3 / 8 = 48
    assert_eq!(m.packed_dim(), 48);
    // num_blocks = head_dim / 32 = 4
    assert_eq!(m.num_blocks(), 4);
}

// -- LayerStorage tests -----------------------------------------------------

#[test]
fn layer_storage_default_is_empty() {
    let layer = LayerStorage::default();
    assert_eq!(layer.seq_len(), 0);
    assert!(!layer.is_active());
    assert_eq!(layer.capacity(), 0);
    assert!(layer.buffers().is_none());
}

#[test]
fn layer_storage_ensure_capacity_allocates_buffers() {
    let m = metadata();
    let mut layer = LayerStorage::default();
    layer.ensure_capacity(4, &m, &Device::Cpu).unwrap();
    // Buffers allocated but not yet marked active (append sets active).
    assert!(!layer.is_active());
    assert!(layer.capacity() >= 4);
    assert!(layer.buffers().is_some());
    // memory_usage reports 0 before any data is appended
    assert_eq!(layer.memory_usage(&m), 0);
}

#[test]
fn layer_storage_append_marks_active_and_updates_seq_len() {
    let m = metadata();
    let packed_dim = m.packed_dim();
    let num_blocks = m.num_blocks();
    let seq = 4;

    let mut layer = LayerStorage::default();
    layer.ensure_capacity(seq, &m, &Device::Cpu).unwrap();

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
    layer.append(0, &kv, seq).unwrap();

    assert!(layer.is_active());
    assert_eq!(layer.seq_len(), seq);
    // memory usage now positive
    assert!(layer.memory_usage(&m) > 0);
}

#[test]
fn layer_storage_validate_rejects_inconsistent_state() {
    // Consistent: default is valid.
    let default_layer = LayerStorage::default();
    default_layer.validate().unwrap();

    // Consistent: populated layer is valid.
    let m = metadata();
    let mut layer = LayerStorage::default();
    layer.ensure_capacity(2, &m, &Device::Cpu).unwrap();
    let ki = Tensor::zeros((NUM_KV_HEADS, 2, m.packed_dim()), DType::U8, &Device::Cpu).unwrap();
    let ks = Tensor::zeros((NUM_KV_HEADS, 2, m.num_blocks()), DType::F16, &Device::Cpu).unwrap();
    let kv = QuantizedKV {
        k_indices: &ki,
        k_scales: &ks,
        v_indices: &ki,
        v_scales: &ks,
    };
    layer.append(0, &kv, 2).unwrap();
    layer.validate().unwrap();
}

#[test]
fn layer_storage_reset_clears_state() {
    let m = metadata();
    let packed_dim = m.packed_dim();
    let num_blocks = m.num_blocks();
    let seq = 2;

    let mut layer = LayerStorage::default();
    layer.ensure_capacity(seq, &m, &Device::Cpu).unwrap();
    let ki = Tensor::zeros((NUM_KV_HEADS, seq, packed_dim), DType::U8, &Device::Cpu).unwrap();
    let ks = Tensor::zeros((NUM_KV_HEADS, seq, num_blocks), DType::F16, &Device::Cpu).unwrap();
    let kv = QuantizedKV {
        k_indices: &ki,
        k_scales: &ks,
        v_indices: &ki,
        v_scales: &ks,
    };
    layer.append(0, &kv, seq).unwrap();
    assert!(layer.is_active());

    layer.reset();
    assert!(!layer.is_active());
    assert_eq!(layer.seq_len(), 0);
    assert!(layer.buffers().is_none());
}

#[test]
fn layer_storage_ensure_capacity_preserves_old_data_on_growth() {
    let m = metadata();
    let packed_dim = m.packed_dim();
    let num_blocks = m.num_blocks();
    let seq = 2;

    let mut layer = LayerStorage::default();
    layer.ensure_capacity(seq, &m, &Device::Cpu).unwrap();

    // Append distinguishable data: ones.
    let ki = Tensor::ones((NUM_KV_HEADS, seq, packed_dim), DType::U8, &Device::Cpu).unwrap();
    let ks = Tensor::ones((NUM_KV_HEADS, seq, num_blocks), DType::F16, &Device::Cpu).unwrap();
    let kv = QuantizedKV {
        k_indices: &ki,
        k_scales: &ks,
        v_indices: &ki,
        v_scales: &ks,
    };
    layer.append(0, &kv, seq).unwrap();

    // Grow capacity.
    layer.ensure_capacity(seq + 100, &m, &Device::Cpu).unwrap();
    assert!(layer.capacity() >= seq + 100);
    // Old data at positions 0..seq should be preserved (all ones).
    let preserved = layer
        .buffers()
        .unwrap()
        .k_indices
        .narrow(1, 0, seq)
        .unwrap()
        .to_vec3::<u8>()
        .unwrap();
    for head in &preserved {
        for row in head {
            for &byte in row {
                assert_eq!(byte, 1, "old data lost after capacity growth");
            }
        }
    }
}

// -- Roundtrip integration test --------------------------------------------

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
    Ok(())
}
