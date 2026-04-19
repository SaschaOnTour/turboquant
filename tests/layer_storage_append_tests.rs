//! LayerStorage append & validate tests.
//!
//! Extracted from the former `cache_storage_tests.rs`.

#![cfg(feature = "candle")]

use candle_core::{DType, Device, Tensor};
use turboquant::cache::{LayerStorage, QuantizedKV, StorageMetadata};

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

#[test]
fn append_marks_active_and_updates_seq_len() {
    let m = metadata();
    let seq = 4;
    let mut layer = LayerStorage::default();
    layer.ensure_capacity(seq, &m, &Device::Cpu).unwrap();

    let indices =
        Tensor::zeros((NUM_KV_HEADS, seq, m.packed_dim()), DType::U8, &Device::Cpu).unwrap();
    let scales = Tensor::zeros(
        (NUM_KV_HEADS, seq, m.num_blocks()),
        DType::F16,
        &Device::Cpu,
    )
    .unwrap();
    let kv = QuantizedKV {
        k_indices: &indices,
        k_scales: &scales,
        v_indices: &indices,
        v_scales: &scales,
    };
    layer.append(0, &kv, seq).unwrap();

    assert!(layer.is_active());
    assert_eq!(layer.seq_len(), seq);
    assert!(layer.memory_usage(&m) > 0);
}

#[test]
fn validate_accepts_consistent_state() {
    LayerStorage::default().validate().unwrap();

    let m = metadata();
    let mut layer = LayerStorage::default();
    layer.ensure_capacity(2, &m, &Device::Cpu).unwrap();
    let indices =
        Tensor::zeros((NUM_KV_HEADS, 2, m.packed_dim()), DType::U8, &Device::Cpu).unwrap();
    let scales =
        Tensor::zeros((NUM_KV_HEADS, 2, m.num_blocks()), DType::F16, &Device::Cpu).unwrap();
    let kv = QuantizedKV {
        k_indices: &indices,
        k_scales: &scales,
        v_indices: &indices,
        v_scales: &scales,
    };
    layer.append(0, &kv, 2).unwrap();
    layer.validate().unwrap();
}
