//! LayerStorage capacity growth preserves data tests.
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
fn ensure_capacity_preserves_old_data_on_growth() {
    let m = metadata();
    let seq = 2;
    let mut layer = LayerStorage::default();
    layer.ensure_capacity(seq, &m, &Device::Cpu).unwrap();

    // Append distinguishable data (all ones).
    let indices =
        Tensor::ones((NUM_KV_HEADS, seq, m.packed_dim()), DType::U8, &Device::Cpu).unwrap();
    let scales = Tensor::ones(
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

    layer.ensure_capacity(seq + 100, &m, &Device::Cpu).unwrap();
    assert!(layer.capacity() >= seq + 100);

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
