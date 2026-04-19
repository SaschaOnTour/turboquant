//! Packed tests extracted from the former `roundtrip_tests.rs`.

// qual:allow(srp) — cohesive integration-test module
use half::f16;
use turboquant::packed::{
    pack_2bit, pack_3bit, pack_4bit, pack_indices_2bit, pack_indices_3bit, pack_indices_4bit,
    unpack_2bit, unpack_3bit, unpack_4bit, unpack_indices_2bit, unpack_indices_3bit,
    unpack_indices_4bit, PackedBlock, TurboQuantConfig,
};

/// Representative residual-norm value used in the size-byte fixtures.
/// The exact magnitude does not matter — the tests only verify byte counts.
const SAMPLE_RESIDUAL_NORM: f32 = 2.5;

// ----- 3-bit roundtrip ---------------------------------------------------

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

// ----- 4-bit roundtrip ---------------------------------------------------

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

// ----- TurboQuantConfig validation --------------------------------------

#[test]
fn config_accepts_bits_2() {
    assert!(TurboQuantConfig::new(2, 64).is_ok());
}

#[test]
fn config_rejects_bits_1() {
    assert!(TurboQuantConfig::new(1, 64).is_err());
}

#[test]
fn config_rejects_bits_5() {
    assert!(TurboQuantConfig::new(5, 64).is_err());
}

#[test]
fn config_rejects_non_power_of_two() {
    assert!(TurboQuantConfig::new(3, 33).is_err());
    assert!(TurboQuantConfig::new(4, 100).is_err());
}

#[test]
fn config_rejects_dim_zero() {
    assert!(TurboQuantConfig::new(3, 0).is_err());
}

#[test]
fn config_accepts_valid_3bit() {
    // Validates that new(3, 64) succeeds -- the config is usable for quantization.
    let _cfg = TurboQuantConfig::new(3, 64).unwrap();
}

#[test]
fn config_accepts_valid_4bit() {
    // Validates that new(4, 256) succeeds -- the config is usable for quantization.
    let _cfg = TurboQuantConfig::new(4, 256).unwrap();
}

// ----- Full vector roundtrip (128 elements) ------------------------------

#[test]
fn full_vector_roundtrip_3bit_128() {
    let indices: Vec<u8> = (0..128).map(|i| (i % 8) as u8).collect();
    let packed = pack_indices_3bit(&indices);
    let unpacked = unpack_indices_3bit(&packed, 128);
    assert_eq!(indices, unpacked);
}

#[test]
fn full_vector_roundtrip_4bit_128() {
    let indices: Vec<u8> = (0..128).map(|i| (i % 16) as u8).collect();
    let packed = pack_indices_4bit(&indices);
    let unpacked = unpack_indices_4bit(&packed, 128);
    assert_eq!(indices, unpacked);
}

// ----- Block roundtrip ---------------------------------------------------

#[test]
fn packed_block_tq3_roundtrip() {
    let indices: Vec<u8> = (0..64).map(|i| (i % 8) as u8).collect();
    let scale = f16::from_f32(3.25);
    let block = PackedBlock::new(3, scale, &indices);
    let recovered = block.unpack(64);
    assert_eq!(indices, recovered);
    assert_eq!(block.scale, scale);
}

#[test]
fn packed_block_tq4_roundtrip() {
    let indices: Vec<u8> = (0..64).map(|i| (i % 16) as u8).collect();
    let scale = f16::from_f32(2.71);
    let block = PackedBlock::new(4, scale, &indices);
    let recovered = block.unpack(64);
    assert_eq!(indices, recovered);
    assert_eq!(block.scale, scale);
}

// ----- 2-bit roundtrip ---------------------------------------------------

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

#[test]
fn packed_block_tq2_roundtrip() {
    let indices: Vec<u8> = (0..64).map(|i| (i % 4) as u8).collect();
    let scale = f16::from_f32(1.23);
    let block = PackedBlock::new(2, scale, &indices);
    let recovered = block.unpack(64);
    assert_eq!(indices, recovered);
    assert_eq!(block.scale, scale);
}

#[test]
fn packed_block_tq2_size_bytes_dim_32() {
    // 32 indices / 4 per byte = 8 bytes packed
    // total = 2 (scale) + 8 = 10
    let indices = vec![0u8; 32];
    let block = PackedBlock::new(2, f16::from_f32(1.0), &indices);
    assert_eq!(block.size_bytes(), 10);
}

#[test]
fn packed_block_tq2_size_bytes_dim_128() {
    // 128 / 4 = 32 bytes packed => total 34
    let indices = vec![1u8; 128];
    let block = PackedBlock::new(2, f16::from_f32(SAMPLE_RESIDUAL_NORM), &indices);
    assert_eq!(block.size_bytes(), 34);
}

#[test]
fn config_accepts_valid_2bit() {
    // Validates that new(2, 64) succeeds -- the config is usable for quantization.
    let _cfg = TurboQuantConfig::new(2, 64).unwrap();
}
