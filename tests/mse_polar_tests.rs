//! PolarQuant roundtrip MSE validation (raw quantize/dequantize).
//!
//! Extracted from the former `mse_validation.rs`.

use turboquant::packed::TurboQuantConfig;
use turboquant::quantize::{dequantize_vec, quantize_vec};
use turboquant::test_utils::random_normal_vec;

/// Rotation seed (shared across MSE tests).
const MSE_SEED: u64 = 42;
/// RNG seed for the input Gaussian samples (d=128 suite).
const MSE_RNG_SEED: u64 = 12345;
/// Number of vectors for the tight d=128 suite.
const MSE_NUM_VECTORS_128: usize = 10_000;
/// Number of vectors for the lighter d=256 suite.
const MSE_NUM_VECTORS_256: usize = 1_000;

/// Expected PolarQuant TQ3 d=128 normalized-MSE range.
const POLAR_TQ3_D128_MIN: f64 = 0.030;
const POLAR_TQ3_D128_MAX: f64 = 0.038;
/// Expected PolarQuant TQ4 d=128 normalized-MSE range.
const POLAR_TQ4_D128_MIN: f64 = 0.007;
const POLAR_TQ4_D128_MAX: f64 = 0.011;
/// Expected PolarQuant TQ3 d=256 normalized-MSE range.
const POLAR_TQ3_D256_MIN: f64 = 0.025;
const POLAR_TQ3_D256_MAX: f64 = 0.040;
/// Expected PolarQuant TQ4 d=256 normalized-MSE range.
const POLAR_TQ4_D256_MIN: f64 = 0.005;
const POLAR_TQ4_D256_MAX: f64 = 0.012;

/// Computes the normalized MSE across `num_vectors` random-normal inputs.
///   MSE = mean( ||x - dequant(quant(x))||² / ||x||² )
fn compute_normalized_mse(bits: u8, dim: usize, num_vectors: usize) -> f64 {
    let config = TurboQuantConfig::new(bits, dim)
        .unwrap()
        .with_seed(MSE_SEED);

    let mut total_nmse = 0.0_f64;
    let mut valid_count = 0usize;

    for i in 0..num_vectors {
        let data = random_normal_vec(dim, MSE_RNG_SEED.wrapping_add(i as u64));
        let norm_sq = data.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>();
        if norm_sq < 1e-8 {
            continue;
        }

        let block = quantize_vec(&config, &data).unwrap();
        let recovered = dequantize_vec(&config, &block).unwrap();

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
fn mse_tq3_d128_in_expected_range() {
    let mse = compute_normalized_mse(3, 128, MSE_NUM_VECTORS_128);
    eprintln!("TQ3 d=128 normalized MSE: {mse:.6}");
    assert!(
        (POLAR_TQ3_D128_MIN..=POLAR_TQ3_D128_MAX).contains(&mse),
        "TQ3 d=128 MSE {mse:.6} outside [{POLAR_TQ3_D128_MIN}, {POLAR_TQ3_D128_MAX}]"
    );
}

#[test]
fn mse_tq4_d128_in_expected_range() {
    let mse = compute_normalized_mse(4, 128, MSE_NUM_VECTORS_128);
    eprintln!("TQ4 d=128 normalized MSE: {mse:.6}");
    assert!(
        (POLAR_TQ4_D128_MIN..=POLAR_TQ4_D128_MAX).contains(&mse),
        "TQ4 d=128 MSE {mse:.6} outside [{POLAR_TQ4_D128_MIN}, {POLAR_TQ4_D128_MAX}]"
    );
}

#[test]
fn mse_tq3_d256_in_expected_range() {
    let mse = compute_normalized_mse(3, 256, MSE_NUM_VECTORS_256);
    eprintln!("TQ3 d=256 normalized MSE: {mse:.6}");
    assert!(
        (POLAR_TQ3_D256_MIN..=POLAR_TQ3_D256_MAX).contains(&mse),
        "TQ3 d=256 MSE {mse:.6} outside [{POLAR_TQ3_D256_MIN}, {POLAR_TQ3_D256_MAX}]"
    );
}

#[test]
fn mse_tq4_d256_in_expected_range() {
    let mse = compute_normalized_mse(4, 256, MSE_NUM_VECTORS_256);
    eprintln!("TQ4 d=256 normalized MSE: {mse:.6}");
    assert!(
        (POLAR_TQ4_D256_MIN..=POLAR_TQ4_D256_MAX).contains(&mse),
        "TQ4 d=256 MSE {mse:.6} outside [{POLAR_TQ4_D256_MIN}, {POLAR_TQ4_D256_MAX}]"
    );
}
