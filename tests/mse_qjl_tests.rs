//! QJL full-roundtrip MSE validation (quantize_with_qjl + polar dequantize).
//!
//! Extracted from the former `mse_validation.rs`.

use turboquant::packed::TurboQuantConfig;
use turboquant::qjl::quantize_with_qjl;
use turboquant::quantize::{dequantize_vec, quantize_vec};
use turboquant::test_utils::random_normal_vec;

/// Rotation seed (shared across MSE tests).
const MSE_SEED: u64 = 42;
/// RNG seed offset for QJL input Gaussians.
const QJL_MSE_RNG_SEED: u64 = 67890;
/// QJL seed base (incremented per vector).
const QJL_MSE_SEED: u64 = 54321;

/// Number of vectors for the QJL suite (lighter than the polar suite).
const QJL_MSE_NUM_VECTORS: usize = 1_000;

/// Expected QJL TQ3 d=128 range (2-bit polar internally).
const QJL_TQ3_D128_MIN: f64 = 0.03;
const QJL_TQ3_D128_MAX: f64 = 0.20;
/// Expected QJL TQ4 d=128 range (3-bit polar internally).
const QJL_TQ4_D128_MIN: f64 = 0.01;
const QJL_TQ4_D128_MAX: f64 = 0.10;
/// Expected QJL TQ3 d=256 range (2-bit polar internally).
const QJL_TQ3_D256_MIN: f64 = 0.03;
const QJL_TQ3_D256_MAX: f64 = 0.20;
/// Expected QJL TQ4 d=256 range (3-bit polar internally).
const QJL_TQ4_D256_MIN: f64 = 0.01;
const QJL_TQ4_D256_MAX: f64 = 0.10;

/// Computes the normalized MSE for the QJL roundtrip: `quantize_with_qjl` →
/// dequantize the inner polar block with `(bits-1)`-bit polar quantization.
fn compute_qjl_roundtrip_mse(bits: u8, dim: usize, num_vectors: usize) -> f64 {
    let config = TurboQuantConfig::new(bits, dim)
        .unwrap()
        .with_seed(MSE_SEED);
    let polar_bits = bits - 1;
    let polar_config = TurboQuantConfig::new(polar_bits, dim)
        .unwrap()
        .with_seed(MSE_SEED);

    let mut total_nmse = 0.0_f64;
    let mut valid_count = 0usize;

    for i in 0..num_vectors {
        let data = random_normal_vec(dim, QJL_MSE_RNG_SEED.wrapping_add(i as u64));
        let norm_sq = data.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>();
        if norm_sq < 1e-8 {
            continue;
        }

        let qjl_seed = QJL_MSE_SEED.wrapping_add(i as u64);
        let _qjl_block = quantize_with_qjl(&config, &data, qjl_seed).unwrap();

        let polar_block = quantize_vec(&polar_config, &data).unwrap();
        let recovered = dequantize_vec(&polar_config, &polar_block).unwrap();

        let err_sq: f64 = data
            .iter()
            .zip(recovered.iter())
            .map(|(&a, &b)| {
                let diff = a as f64 - b as f64;
                diff * diff
            })
            .sum();

        total_nmse += err_sq / norm_sq;
        valid_count += 1;
    }

    total_nmse / valid_count as f64
}

#[test]
fn qjl_roundtrip_mse_tq3_d128_in_expected_range() {
    let mse = compute_qjl_roundtrip_mse(3, 128, QJL_MSE_NUM_VECTORS);
    eprintln!("QJL TQ3 d=128 polar roundtrip MSE: {mse:.6}");
    assert!(
        (QJL_TQ3_D128_MIN..=QJL_TQ3_D128_MAX).contains(&mse),
        "QJL TQ3 d=128 MSE {mse:.6} outside [{QJL_TQ3_D128_MIN}, {QJL_TQ3_D128_MAX}]"
    );
}

#[test]
fn qjl_roundtrip_mse_tq4_d128_in_expected_range() {
    let mse = compute_qjl_roundtrip_mse(4, 128, QJL_MSE_NUM_VECTORS);
    eprintln!("QJL TQ4 d=128 polar roundtrip MSE: {mse:.6}");
    assert!(
        (QJL_TQ4_D128_MIN..=QJL_TQ4_D128_MAX).contains(&mse),
        "QJL TQ4 d=128 MSE {mse:.6} outside [{QJL_TQ4_D128_MIN}, {QJL_TQ4_D128_MAX}]"
    );
}

#[test]
fn qjl_roundtrip_mse_tq3_d256_in_expected_range() {
    let mse = compute_qjl_roundtrip_mse(3, 256, QJL_MSE_NUM_VECTORS);
    eprintln!("QJL TQ3 d=256 polar roundtrip MSE: {mse:.6}");
    assert!(
        (QJL_TQ3_D256_MIN..=QJL_TQ3_D256_MAX).contains(&mse),
        "QJL TQ3 d=256 MSE {mse:.6} outside [{QJL_TQ3_D256_MIN}, {QJL_TQ3_D256_MAX}]"
    );
}

#[test]
fn qjl_roundtrip_mse_tq4_d256_in_expected_range() {
    let mse = compute_qjl_roundtrip_mse(4, 256, QJL_MSE_NUM_VECTORS);
    eprintln!("QJL TQ4 d=256 polar roundtrip MSE: {mse:.6}");
    assert!(
        (QJL_TQ4_D256_MIN..=QJL_TQ4_D256_MAX).contains(&mse),
        "QJL TQ4 d=256 MSE {mse:.6} outside [{QJL_TQ4_D256_MIN}, {QJL_TQ4_D256_MAX}]"
    );
}
