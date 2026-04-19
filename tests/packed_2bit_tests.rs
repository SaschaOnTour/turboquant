//! 2-bit packing roundtrip tests: pack_2bit + unpack_2bit are inverses.
//!
//! Extracted from the former `packed_tests.rs`.

use turboquant::packed::{pack_2bit, pack_indices_2bit, unpack_2bit, unpack_indices_2bit};

#[test]
fn roundtrip_2bit_all_valid_values() {
    for a in 0u8..=3 {
        for b in 0u8..=3 {
            for c in 0u8..=3 {
                for d in 0u8..=3 {
                    let values: [u8; 4] = [a, b, c, d];
                    let packed = pack_2bit(&values);
                    let unpacked = unpack_2bit(packed);
                    assert_eq!(values, unpacked, "failed for a={a}, b={b}, c={c}, d={d}");
                }
            }
        }
    }
}

#[test]
fn roundtrip_2bit_all_zeros() {
    let values = [0u8; 4];
    assert_eq!(unpack_2bit(pack_2bit(&values)), values);
}

#[test]
fn roundtrip_2bit_all_max() {
    let values = [3u8; 4];
    assert_eq!(unpack_2bit(pack_2bit(&values)), values);
}

#[test]
fn full_vector_roundtrip_2bit_128() {
    let indices: Vec<u8> = (0..128).map(|i| (i % 4) as u8).collect();
    let packed = pack_indices_2bit(&indices);
    let unpacked = unpack_indices_2bit(&packed, 128);
    assert_eq!(indices, unpacked);
}
