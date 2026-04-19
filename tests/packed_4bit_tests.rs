//! 4-bit packing roundtrip tests: pack_4bit + unpack_4bit are inverses.
//!
//! Extracted from the former `packed_tests.rs`.

use turboquant::packed::{pack_4bit, pack_indices_4bit, unpack_4bit, unpack_indices_4bit};

#[test]
fn roundtrip_4bit_all_valid_values() {
    for a in 0u8..=15 {
        for b in 0u8..=15 {
            let values: [u8; 2] = [a, b];
            let packed = pack_4bit(&values);
            let unpacked = unpack_4bit(packed);
            assert_eq!(values, unpacked, "failed for a={a}, b={b}");
        }
    }
}

#[test]
fn roundtrip_4bit_all_zeros() {
    let values = [0u8; 2];
    assert_eq!(unpack_4bit(pack_4bit(&values)), values);
}

#[test]
fn roundtrip_4bit_all_max() {
    let values = [15u8; 2];
    assert_eq!(unpack_4bit(pack_4bit(&values)), values);
}

#[test]
fn roundtrip_4bit_mixed() {
    let values: [u8; 2] = [3, 12];
    assert_eq!(unpack_4bit(pack_4bit(&values)), values);
}

#[test]
fn full_vector_roundtrip_4bit_128() {
    let indices: Vec<u8> = (0..128).map(|i| (i % 16) as u8).collect();
    let packed = pack_indices_4bit(&indices);
    let unpacked = unpack_indices_4bit(&packed, 128);
    assert_eq!(indices, unpacked);
}
