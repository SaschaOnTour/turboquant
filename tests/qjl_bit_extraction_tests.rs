//! QJL sign-bit extraction reference test (tensor ops vs bitwise AND).
//!
//! Extracted from the former `cache_type_correctness.rs`.

#![cfg(feature = "candle")]

use candle_core::{DType, Device, Tensor};

/// Verify that the tensor-based bit extraction used by `unpack_qjl_signs`
/// produces the same signs as a direct Rust bitwise AND implementation.
#[test]
fn bit_extraction_matches_reference() {
    let test_bytes: Vec<u8> = vec![
        0b00000000, 0b11111111, 0b10101010, 0b01010101, 0b00000001, 0b10000000, 0b11001100,
        0b00110011,
    ];
    let dim = test_bytes.len() * 8;

    let mut reference_signs = Vec::with_capacity(dim);
    for &byte in &test_bytes {
        for bit in 0..8u8 {
            let is_set = (byte & (1 << bit)) != 0;
            reference_signs.push(if is_set { 1.0f32 } else { -1.0f32 });
        }
    }

    let byte_tensor = Tensor::from_vec(test_bytes, (1, 8), &Device::Cpu).unwrap();
    let bit_masks =
        Tensor::from_vec(vec![1u8, 2, 4, 8, 16, 32, 64, 128], (1, 1, 8), &Device::Cpu).unwrap();

    let signs_u8 = byte_tensor.unsqueeze(2).unwrap();
    let bytes_f = signs_u8.to_dtype(DType::F32).unwrap();
    let masks_f = bit_masks.to_dtype(DType::F32).unwrap();
    let divided = bytes_f.broadcast_div(&masks_f).unwrap().floor().unwrap();
    let bit_set = ((&divided / 2.0).unwrap().floor().unwrap() * 2.0 - &divided)
        .unwrap()
        .abs()
        .unwrap();
    let signs_float = ((bit_set * 2.0).unwrap() - 1.0)
        .unwrap()
        .reshape((1, dim))
        .unwrap();

    let tensor_signs: Vec<f32> = signs_float.flatten_all().unwrap().to_vec1().unwrap();

    assert_eq!(
        tensor_signs.len(),
        reference_signs.len(),
        "length mismatch: tensor={} vs reference={}",
        tensor_signs.len(),
        reference_signs.len()
    );
    for (i, (t, r)) in tensor_signs.iter().zip(reference_signs.iter()).enumerate() {
        assert_eq!(
            t,
            r,
            "bit {i} mismatch: tensor={t}, reference={r} (byte={}, bit={})",
            i / 8,
            i % 8
        );
    }
}
