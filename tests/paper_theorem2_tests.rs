//! Paper verification: Theorem 2 — unbiasedness & distortion bound
//!
//! Verifies the TurboQuant paper (Zandieh et al., ICLR 2026) against
//! the implementation. Extracted from the former
//! `paper_verification_tests.rs`.

use turboquant::packed::TurboQuantConfig;
use turboquant::qjl::{dot_product, estimate_inner_product_single, quantize_with_qjl};
use turboquant::quantize::{dequantize_vec, quantize_vec};
use turboquant::test_utils::random_unit_vec;

/// Test dimension (power of two for WHT).
const DIM: usize = 128;
/// Rotation seed.
const ROTATION_SEED: u64 = 42;
/// Number of samples for statistical tests.
const STAT_SAMPLES: usize = 2000;

// Paper Theorem 2 distortion coefficients.
const DISTORTION_COEFF_B3: f64 = 0.18;
const DISTORTION_COEFF_B4: f64 = 0.047;

// Per-test distinct prime seed multipliers.
const SEED_PRIME_UNBIAS_X: u64 = 31;
const SEED_PRIME_UNBIAS_Y: u64 = 37;
const SEED_PRIME_DISTORTION_B3_X: u64 = 43;
const SEED_PRIME_DISTORTION_B3_Y: u64 = 47;
const SEED_PRIME_DISTORTION_B4_X: u64 = 53;
const SEED_PRIME_DISTORTION_B4_Y: u64 = 59;
const SEED_PRIME_POLAR_X: u64 = 61;
const SEED_PRIME_POLAR_Y: u64 = 67;
const QJL_SEED_OFFSET_B4: u64 = 77_777;
const UNBIAS_MEAN_TOLERANCE: f64 = 0.03;

/// Paper Theorem 2: TurboQuantprod inner product estimate is unbiased.
///
/// For each sample: generate random x, y on S^{d-1}, quantize x with a
/// DIFFERENT QJL seed (= different S), estimate ⟨y, x̃⟩, measure bias.
/// Over many seeds: E[⟨y, x̃⟩] should equal ⟨y, x⟩.
#[test]
fn theorem2_unbiasedness() {
    let total_bits: u8 = 3; // TQ3: 2-bit polar + 1-bit QJL

    let mut bias_sum = 0.0_f64;

    for i in 0..STAT_SAMPLES {
        // CRITICAL: different S per sample (paper's expectation is over S)
        let qjl_seed = 12345_u64.wrapping_add(i as u64);

        let x = random_unit_vec(DIM, i as u64 * SEED_PRIME_UNBIAS_X + 1000);
        let y = random_unit_vec(DIM, i as u64 * SEED_PRIME_UNBIAS_Y + 2000);
        let true_ip = dot_product(&x, &y) as f64;

        let config = TurboQuantConfig::new(total_bits, DIM)
            .unwrap()
            .with_seed(ROTATION_SEED);
        let block = quantize_with_qjl(&config, &x, qjl_seed).unwrap();
        let est = estimate_inner_product_single(&y, &block, &config, qjl_seed).unwrap() as f64;

        bias_sum += est - true_ip;
    }

    let mean_bias = (bias_sum / STAT_SAMPLES as f64).abs();

    // Paper: exact unbiasedness. With 2000 samples, tolerance for statistical noise.
    let tolerance = UNBIAS_MEAN_TOLERANCE;
    assert!(
        mean_bias < tolerance,
        "Paper Theorem 2 violated: mean bias = {mean_bias:.4} \
         (expected < {tolerance}) over {STAT_SAMPLES} samples. \
         E[⟨y, x̃⟩] should equal ⟨y, x⟩."
    );
}

/// Per-bit-width parameters for the distortion sweep.
struct DistortionCase {
    total_bits: u8,
    approx_coeff: f64,
    qjl_seed_base: u64,
    seed_prime_x: u64,
    seed_x_offset: u64,
    seed_prime_y: u64,
    seed_y_offset: u64,
}

const DISTORTION_CASE_B3: DistortionCase = DistortionCase {
    total_bits: 3,
    approx_coeff: DISTORTION_COEFF_B3,
    qjl_seed_base: 99_999,
    seed_prime_x: SEED_PRIME_DISTORTION_B3_X,
    seed_x_offset: 3000,
    seed_prime_y: SEED_PRIME_DISTORTION_B3_Y,
    seed_y_offset: 4000,
};

const DISTORTION_CASE_B4: DistortionCase = DistortionCase {
    total_bits: 4,
    approx_coeff: DISTORTION_COEFF_B4,
    qjl_seed_base: QJL_SEED_OFFSET_B4,
    seed_prime_x: SEED_PRIME_DISTORTION_B4_X,
    seed_x_offset: 5000,
    seed_prime_y: SEED_PRIME_DISTORTION_B4_Y,
    seed_y_offset: 6000,
};

/// Margin applied to the paper's general bound to absorb Monte-Carlo noise.
const DISTORTION_TEST_MARGIN: f64 = 2.0;

/// Paper Theorem 2: inner product distortion is bounded for b=3 and b=4.
///
/// Loops over both bit widths in a single test: each b has its own seeds
/// and QJL-seed base, but the assertion logic is identical. Empirical
/// distortion must stay below 2× the general paper bound.
#[test]
fn theorem2_distortion_bounds() {
    for case in [&DISTORTION_CASE_B3, &DISTORTION_CASE_B4] {
        let general_bound = 3.0 * std::f64::consts::PI.powi(2)
            / (DIM as f64 * 4.0_f64.powi(case.total_bits as i32));
        let approximate_value = case.approx_coeff / DIM as f64;
        let config = TurboQuantConfig::new(case.total_bits, DIM)
            .unwrap()
            .with_seed(ROTATION_SEED);

        let mut distortion_sum = 0.0_f64;
        for i in 0..STAT_SAMPLES {
            let qjl_seed = case.qjl_seed_base.wrapping_add(i as u64);
            let x = random_unit_vec(DIM, i as u64 * case.seed_prime_x + case.seed_x_offset);
            let y = random_unit_vec(DIM, i as u64 * case.seed_prime_y + case.seed_y_offset);
            let true_ip = dot_product(&x, &y) as f64;

            let block = quantize_with_qjl(&config, &x, qjl_seed).unwrap();
            let est = estimate_inner_product_single(&y, &block, &config, qjl_seed).unwrap() as f64;
            distortion_sum += (true_ip - est).powi(2);
        }

        let empirical_distortion = distortion_sum / STAT_SAMPLES as f64;
        let bits = case.total_bits;
        eprintln!(
            "Theorem 2 distortion (b={bits}, d={DIM}): \
             empirical={empirical_distortion:.6}, paper_approx={approximate_value:.6}, \
             general_bound={general_bound:.6}"
        );
        let test_bound = general_bound * DISTORTION_TEST_MARGIN;
        assert!(
            empirical_distortion < test_bound,
            "Paper Theorem 2 distortion bound violated (b={bits}): \
             empirical={empirical_distortion:.6} > {DISTORTION_TEST_MARGIN}×bound={test_bound:.6}"
        );
    }
}

/// Paper Section 3.2: PolarQuant without QJL has multiplicative bias.
///
/// The 2/π bias is MULTIPLICATIVE: E[⟨y, x̃_mse⟩] = α·⟨y, x⟩ where α < 1.
/// For random unit vectors, E[⟨y,x⟩] = 0, so the additive bias is zero.
/// We detect the multiplicative bias by measuring the SLOPE of
/// polar_estimate vs true_ip (should be < 1.0 for polar, = 1.0 for QJL).
///
/// Equivalently: E[polar_ip · true_ip] / E[true_ip²] < 1.0
// qual:allow(complexity) — one statistical assertion per test; splitting would require duplicating the 2000-sample Monte-Carlo loop
#[test]
fn polar_only_has_multiplicative_bias_qjl_fixes_it() {
    let total_bits: u8 = 3;
    let polar_bits = total_bits - 1;

    let mut polar_xy_sum = 0.0_f64; // Σ polar_ip × true_ip
    let mut qjl_xy_sum = 0.0_f64; // Σ qjl_ip × true_ip
    let mut true_sq_sum = 0.0_f64; // Σ true_ip²

    for i in 0..STAT_SAMPLES {
        let qjl_seed = 55555_u64.wrapping_add(i as u64);

        let x = random_unit_vec(DIM, i as u64 * SEED_PRIME_POLAR_X + 7000);
        let y = random_unit_vec(DIM, i as u64 * SEED_PRIME_POLAR_Y + 8000);
        let true_ip = dot_product(&x, &y) as f64;

        // Polar-only (no QJL)
        let polar_config = TurboQuantConfig::new(polar_bits, DIM)
            .unwrap()
            .with_seed(ROTATION_SEED);
        let polar_block = quantize_vec(&polar_config, &x).unwrap();
        let reconstructed = dequantize_vec(&polar_config, &polar_block).unwrap();
        let polar_ip = dot_product(&y, &reconstructed) as f64;

        // With QJL
        let config = TurboQuantConfig::new(total_bits, DIM)
            .unwrap()
            .with_seed(ROTATION_SEED);
        let block = quantize_with_qjl(&config, &x, qjl_seed).unwrap();
        let qjl_ip = estimate_inner_product_single(&y, &block, &config, qjl_seed).unwrap() as f64;

        polar_xy_sum += polar_ip * true_ip;
        qjl_xy_sum += qjl_ip * true_ip;
        true_sq_sum += true_ip * true_ip;
    }

    // Regression slope: E[est·true] / E[true²]
    // For unbiased estimator: slope = 1.0
    // For multiplicatively biased (α): slope = α < 1.0
    let polar_slope = polar_xy_sum / true_sq_sum;
    let qjl_slope = qjl_xy_sum / true_sq_sum;

    eprintln!(
        "Polar-only slope: {polar_slope:.4} (should be < 1.0, ≈ 2/π = {:.4} for b=1), \
         QJL slope: {qjl_slope:.4} (should ≈ 1.0)",
        2.0 / std::f64::consts::PI
    );

    // Polar-only MUST have multiplicative bias (slope < 1.0)
    assert!(
        polar_slope < 0.99,
        "Polar-only should have multiplicative bias (slope < 1), got {polar_slope:.4}"
    );

    // QJL should fix the multiplicative bias (slope ≈ 1.0)
    assert!(
        qjl_slope > 0.95 && qjl_slope < 1.05,
        "QJL slope should be ≈ 1.0, got {qjl_slope:.4}"
    );

    // QJL slope should be closer to 1.0 than polar slope
    assert!(
        (qjl_slope - 1.0).abs() < (polar_slope - 1.0).abs(),
        "QJL should be closer to unbiased: |qjl-1|={:.4} vs |polar-1|={:.4}",
        (qjl_slope - 1.0).abs(),
        (polar_slope - 1.0).abs()
    );
}
