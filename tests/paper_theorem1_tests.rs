//! Paper verification: Theorem 1 — PolarQuant MSE bound
//!
//! Verifies the TurboQuant paper (Zandieh et al., ICLR 2026) against
//! the implementation. Extracted from the former
//! `paper_verification_tests.rs`.

use turboquant::packed::TurboQuantConfig;
use turboquant::quantize::{dequantize_vec, quantize_vec};
use turboquant::test_utils::random_unit_vec;

/// Test dimension (power of two for WHT).
const DIM: usize = 128;
/// Rotation seed.
const ROTATION_SEED: u64 = 42;
/// Number of samples for statistical tests.
const STAT_SAMPLES: usize = 2000;

// Paper Theorem 1 MSE coefficients.
const MSE_COEFF_B2: f64 = 0.117;
const MSE_COEFF_B3: f64 = 0.030;
const MSE_COEFF_B4: f64 = 0.009;
/// Per-sample seed multiplier (prime) to derive distinct deterministic seeds.
const SEED_PRIME_MSE: u64 = 41;
/// Multiplicative margin applied to the paper's MSE-bound predictions.
const MSE_BOUND_MARGIN: f64 = 1.3;

/// Paper Theorem 1: PolarQuant MSE matches predicted values.
#[test]
fn theorem1_mse_bound() {
    for (bits, expected_mse) in [(2u8, MSE_COEFF_B2), (3, MSE_COEFF_B3), (4, MSE_COEFF_B4)] {
        let config = TurboQuantConfig::new(bits, DIM)
            .unwrap()
            .with_seed(ROTATION_SEED);

        let mut mse_sum = 0.0_f64;
        for i in 0..STAT_SAMPLES {
            let x = random_unit_vec(DIM, i as u64 * SEED_PRIME_MSE + bits as u64 * 10000);
            let block = quantize_vec(&config, &x).unwrap();
            let x_hat = dequantize_vec(&config, &block).unwrap();

            let mse: f64 = x
                .iter()
                .zip(x_hat.iter())
                .map(|(a, b)| ((*a - *b) as f64).powi(2))
                .sum();
            mse_sum += mse;
        }

        let empirical_mse = mse_sum / STAT_SAMPLES as f64;

        // Allow 30% margin: the paper values are approximations, and
        // Rademacher rotation (vs Gaussian in paper) may give slightly
        // different constants.
        let margin = MSE_BOUND_MARGIN;
        eprintln!(
            "Theorem 1 MSE (b={bits}, d={DIM}): empirical={empirical_mse:.6}, \
             paper={expected_mse:.6}, ratio={:.2}",
            empirical_mse / expected_mse
        );

        assert!(
            empirical_mse < expected_mse * margin,
            "Paper Theorem 1 MSE bound violated (b={bits}): \
             empirical={empirical_mse:.6} > {margin}× paper={expected_mse:.6}"
        );
    }
}
