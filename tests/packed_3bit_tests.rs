//! 3-bit packing roundtrip tests: pack_3bit + unpack_3bit are inverses.
//!
//! Extracted from the former `packed_tests.rs`.

use turboquant::packed::{pack_3bit, pack_indices_3bit, unpack_3bit, unpack_indices_3bit};

#[test]
fn roundtrip_3bit_all_valid_values() {
    // Every combination of 0..=7 in the first two slots, fixed elsewhere.
    for a in 0u8..=7 {
        for b in 0u8..=7 {
            let values: [u8; 8] = [a, b, 0, 7, 3, 5, 1, 6];
            let packed = pack_3bit(&values);
            let unpacked = unpack_3bit(&packed);
            assert_eq!(values, unpacked, "failed for a={a}, b={b}");
        }
    }
}

#[test]
fn roundtrip_3bit_all_zeros() {
    let values = [0u8; 8];
    assert_eq!(unpack_3bit(&pack_3bit(&values)), values);
}

#[test]
fn roundtrip_3bit_all_max() {
    let values = [7u8; 8];
    assert_eq!(unpack_3bit(&pack_3bit(&values)), values);
}

#[test]
fn roundtrip_3bit_mixed() {
    let values: [u8; 8] = [1, 3, 5, 7, 0, 2, 4, 6];
    assert_eq!(unpack_3bit(&pack_3bit(&values)), values);
}

#[test]
fn full_vector_roundtrip_3bit_128() {
    let indices: Vec<u8> = (0..128).map(|i| (i % 8) as u8).collect();
    let packed = pack_indices_3bit(&indices);
    let unpacked = unpack_indices_3bit(&packed, 128);
    assert_eq!(indices, unpacked);
}
