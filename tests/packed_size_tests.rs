//! Packed size-in-bytes tests: PackedBlock::size_bytes matches the
//! packed layout (scale bytes + bit-packed indices bytes).
//!
//! Extracted from `packed_tests.rs`.

use half::f16;
use turboquant::packed::PackedBlock;

/// Representative residual-norm value used in the size-byte fixtures.
/// The exact magnitude does not matter — the tests only verify byte counts.
const SAMPLE_RESIDUAL_NORM: f32 = 2.5;

// ----- size_bytes --------------------------------------------------------

#[test]
fn packed_block_tq3_size_bytes_dim_32() {
    // 32 indices / 8 per group = 4 groups * 3 bytes = 12 bytes packed
    // total = 2 (scale) + 12 = 14
    let indices = vec![0u8; 32];
    let block = PackedBlock::new(3, f16::from_f32(1.0), &indices);
    assert_eq!(block.size_bytes(), 14);
}

#[test]
fn packed_block_tq3_size_bytes_dim_128() {
    // 128 / 8 = 16 groups * 3 = 48 bytes packed => total 50
    let indices = vec![3u8; 128];
    let block = PackedBlock::new(3, f16::from_f32(SAMPLE_RESIDUAL_NORM), &indices);
    assert_eq!(block.size_bytes(), 50);
}

#[test]
fn packed_block_tq4_size_bytes_dim_32() {
    // 32 indices / 2 = 16 bytes packed => total 18
    let indices = vec![0u8; 32];
    let block = PackedBlock::new(4, f16::from_f32(1.0), &indices);
    assert_eq!(block.size_bytes(), 18);
}

#[test]
fn packed_block_tq4_size_bytes_dim_128() {
    // 128 / 2 = 64 bytes packed => total 66
    let indices = vec![9u8; 128];
    let block = PackedBlock::new(4, f16::from_f32(0.5), &indices);
    assert_eq!(block.size_bytes(), 66);
}

#[test]
fn packed_block_tq2_size_bytes_dim_128() {
    // 128 / 4 = 32 bytes packed => total 34
    let indices = vec![1u8; 128];
    let block = PackedBlock::new(2, f16::from_f32(SAMPLE_RESIDUAL_NORM), &indices);
    assert_eq!(block.size_bytes(), 34);
}
