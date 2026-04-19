//! LayerStorage init & capacity allocation tests.
//!
//! Extracted from the former `cache_storage_tests.rs`.

#![cfg(feature = "candle")]

use candle_core::Device;
use turboquant::cache::{LayerStorage, StorageMetadata};

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
fn default_is_empty() {
    let layer = LayerStorage::default();
    assert_eq!(layer.seq_len(), 0);
    assert!(!layer.is_active());
    assert_eq!(layer.capacity(), 0);
    assert!(layer.buffers().is_none());
}

#[test]
fn ensure_capacity_allocates_buffers() {
    let m = metadata();
    let mut layer = LayerStorage::default();
    layer.ensure_capacity(4, &m, &Device::Cpu).unwrap();
    assert!(!layer.is_active());
    assert!(layer.capacity() >= 4);
    assert!(layer.buffers().is_some());
    assert_eq!(layer.memory_usage(&m), 0);
}
