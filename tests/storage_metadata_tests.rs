//! Unit tests for `StorageMetadata` — derived packing parameters.
//!
//! Extracted from the former `cache_storage_tests.rs`.

#![cfg(feature = "candle")]

use turboquant::cache::StorageMetadata;

const HEAD_DIM: usize = 128;
const NUM_KV_HEADS: usize = 4;
const BITS: u8 = 3;

#[test]
fn derives_packing_params() {
    let m = StorageMetadata {
        num_kv_heads: NUM_KV_HEADS,
        head_dim: HEAD_DIM,
        bits: BITS,
    };
    // packed_dim = head_dim * bits / 8 = 128 * 3 / 8 = 48
    assert_eq!(m.packed_dim(), 48);
    // num_blocks = head_dim / 32 = 4
    assert_eq!(m.num_blocks(), 4);
}
