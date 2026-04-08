//! Verification tests against TurboQuant paper (Zandieh et al., ICLR 2026).
//!
//! arXiv:2504.19874 — "TurboQuant: Online Vector Quantization with
//! Near-optimal Distortion Rate"
//!
//! These tests verify the implementation against the paper's mathematical
//! guarantees (Theorem 2), NOT against turboquant-rs internals. If
//! turboquant-rs has a bug, these tests should catch it.
//!
//! IMPORTANT: Unit vectors must be sampled uniformly on S^{d-1} via the
//! Gaussian method (d i.i.d. N(0,1) coordinates, then normalize). Using
//! LCG-based uniform coordinates with normalization does NOT produce
//! uniform unit vectors and leads to inflated MSE (see analysis 2026-04-05).

use turboquant::packed::TurboQuantConfig;
use turboquant::qjl::{
    dot_product, estimate_inner_product_single, qjl_scaling_constant, quantize_with_qjl, sign_bit,
};
use turboquant::quantize::{dequantize_vec, quantize_vec};
use turboquant::rotation::wht_inplace;

// ---------------------------------------------------------------------------
// Constants from the paper
// ---------------------------------------------------------------------------

/// Paper Theorem 1: D_mse ≈ C_mse(b) for unit vectors on S^{d-1}, where:
/// b=2: 0.117, b=3: 0.03, b=4: 0.009
const MSE_COEFF_B2: f64 = 0.117;
const MSE_COEFF_B3: f64 = 0.030;
const MSE_COEFF_B4: f64 = 0.009;

/// Paper Theorem 2: D_prod ≈ C_prod(b)/d for unit vectors, where C_prod(b) is:
/// b=1: 1.57, b=2: 0.56, b=3: 0.18, b=4: 0.047
const DISTORTION_COEFF_B3: f64 = 0.18;
const DISTORTION_COEFF_B4: f64 = 0.047;

/// Paper Algorithm 2: QJL dequant scaling factor √(π/2).
const SQRT_PI_OVER_2: f64 = 1.253_314_137_315_500_3;

/// Test dimension (power of two for WHT).
const DIM: usize = 128;

/// Rotation seed.
const ROTATION_SEED: u64 = 42;

/// Number of samples for statistical tests.
const STAT_SAMPLES: usize = 2000;

// ---------------------------------------------------------------------------
// PRNG: SplitMix64 — high-quality 64-bit generator
//
// Same finalizer used by turboquant-rs for Rademacher signs.
// Produces full 64-bit output suitable for Box-Muller transform.
// ---------------------------------------------------------------------------

/// SplitMix64 constants (Stafford variant 13).
const SPLITMIX_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;
const SPLITMIX_MUL1: u64 = 0xbf58_476d_1ce4_e5b9;
const SPLITMIX_MUL2: u64 = 0x94d0_49bb_1331_11eb;

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(SPLITMIX_GAMMA);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(SPLITMIX_MUL1);
        z = (z ^ (z >> 27)).wrapping_mul(SPLITMIX_MUL2);
        z ^ (z >> 31)
    }

    /// Returns a f64 in (0, 1), never exactly 0 or 1.
    fn next_open01(&mut self) -> f64 {
        // Use 53 bits for double precision: (bits >> 11) * 2^-53
        // Add 0.5 ULP to avoid exactly 0.0
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
}

// ---------------------------------------------------------------------------
// Uniform unit vector sampling via Gaussian method
//
// Paper assumes x ∈ S^{d-1} (unit sphere). The standard method:
// 1. Generate d i.i.d. N(0,1) coordinates via Box-Muller
// 2. Normalize to unit length
// This produces vectors uniformly distributed on S^{d-1}.
// ---------------------------------------------------------------------------

/// Generates a deterministic unit vector uniformly on S^{d-1}.
///
/// Uses Box-Muller with SplitMix64 PRNG — produces proper Gaussian
/// coordinates, unlike LCG-uniform normalization which has inflated tails.
fn random_unit_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    let mut gaussians = Vec::with_capacity(dim);

    // Box-Muller: generate pairs of N(0,1) variates
    let pairs = dim.div_ceil(2);
    for _ in 0..pairs {
        let u1 = rng.next_open01();
        let u2 = rng.next_open01();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        gaussians.push(r * theta.cos());
        gaussians.push(r * theta.sin());
    }
    gaussians.truncate(dim);

    // Normalize to unit sphere
    let norm: f64 = gaussians.iter().map(|x| x * x).sum::<f64>().sqrt();
    gaussians.iter().map(|x| (*x / norm) as f32).collect()
}

/// Deterministic pseudo-random vector (unnormalized, for WHT test only).
fn pseudo_random_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    (0..dim)
        .map(|_| (rng.next_u64() as i64) as f32 / (i64::MAX as f32))
        .collect()
}

// ---------------------------------------------------------------------------
// Theorem 2, Claim 1: Unbiasedness
//
// Paper: "Expected inner-product E_x̃[⟨y, x̃⟩] = ⟨y, x⟩"
//
// The expectation is over the randomness of S (the projection matrix).
// We test this by varying the QJL seed (which changes S) across samples.
// ---------------------------------------------------------------------------

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

        let x = random_unit_vec(DIM, i as u64 * 31 + 1000);
        let y = random_unit_vec(DIM, i as u64 * 37 + 2000);
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
    let tolerance = 0.03;
    assert!(
        mean_bias < tolerance,
        "Paper Theorem 2 violated: mean bias = {mean_bias:.4} \
         (expected < {tolerance}) over {STAT_SAMPLES} samples. \
         E[⟨y, x̃⟩] should equal ⟨y, x⟩."
    );
}

// ---------------------------------------------------------------------------
// Theorem 1: MSE bound (PolarQuant)
//
// Paper Theorem 1: D_mse = E[||x - x̃_mse||²] ≈ C_mse(b)
// For b=2: 0.117, b=3: 0.03, b=4: 0.009
//
// This verifies the polar quantizer independently of QJL.
// ---------------------------------------------------------------------------

/// Paper Theorem 1: PolarQuant MSE matches predicted values.
#[test]
fn theorem1_mse_bound() {
    for (bits, expected_mse) in [(2u8, MSE_COEFF_B2), (3, MSE_COEFF_B3), (4, MSE_COEFF_B4)] {
        let config = TurboQuantConfig::new(bits, DIM)
            .unwrap()
            .with_seed(ROTATION_SEED);

        let mut mse_sum = 0.0_f64;
        for i in 0..STAT_SAMPLES {
            let x = random_unit_vec(DIM, i as u64 * 41 + bits as u64 * 10000);
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
        let margin = 1.3;
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

// ---------------------------------------------------------------------------
// Theorem 2, Claim 2: Inner product distortion bound
//
// Paper: D_prod := E[|⟨y,x⟩ - ⟨y,x̃⟩|²] ≤ 3π²·‖y‖² / (d·4^b)
// For b=3, unit vectors: D_prod ≈ 0.18/d
// For b=4, unit vectors: D_prod ≈ 0.047/d
// ---------------------------------------------------------------------------

/// Paper Theorem 2: inner product distortion is bounded.
#[test]
fn theorem2_distortion_bound_b3() {
    let total_bits: u8 = 3;
    let general_bound =
        3.0 * std::f64::consts::PI.powi(2) / (DIM as f64 * 4.0_f64.powi(total_bits as i32));
    let approximate_value = DISTORTION_COEFF_B3 / DIM as f64;

    let mut distortion_sum = 0.0_f64;

    for i in 0..STAT_SAMPLES {
        let qjl_seed = 99999_u64.wrapping_add(i as u64);

        let x = random_unit_vec(DIM, i as u64 * 43 + 3000);
        let y = random_unit_vec(DIM, i as u64 * 47 + 4000);
        let true_ip = dot_product(&x, &y) as f64;

        let config = TurboQuantConfig::new(total_bits, DIM)
            .unwrap()
            .with_seed(ROTATION_SEED);
        let block = quantize_with_qjl(&config, &x, qjl_seed).unwrap();
        let est = estimate_inner_product_single(&y, &block, &config, qjl_seed).unwrap() as f64;

        distortion_sum += (true_ip - est).powi(2);
    }

    let empirical_distortion = distortion_sum / STAT_SAMPLES as f64;

    eprintln!(
        "Theorem 2 distortion (b=3, d={DIM}): empirical={empirical_distortion:.6}, \
         paper_approx={approximate_value:.6}, general_bound={general_bound:.6}"
    );

    // Allow 2x margin over general bound for finite-sample variance
    let test_bound = general_bound * 2.0;
    assert!(
        empirical_distortion < test_bound,
        "Paper Theorem 2 distortion bound violated (b=3): \
         empirical={empirical_distortion:.6} > 2×bound={test_bound:.6}"
    );
}

/// Paper Theorem 2 distortion for b=4.
#[test]
fn theorem2_distortion_bound_b4() {
    let total_bits: u8 = 4;
    let general_bound =
        3.0 * std::f64::consts::PI.powi(2) / (DIM as f64 * 4.0_f64.powi(total_bits as i32));
    let approximate_value = DISTORTION_COEFF_B4 / DIM as f64;

    let mut distortion_sum = 0.0_f64;

    for i in 0..STAT_SAMPLES {
        let qjl_seed = 77777_u64.wrapping_add(i as u64);

        let x = random_unit_vec(DIM, i as u64 * 53 + 5000);
        let y = random_unit_vec(DIM, i as u64 * 59 + 6000);
        let true_ip = dot_product(&x, &y) as f64;

        let config = TurboQuantConfig::new(total_bits, DIM)
            .unwrap()
            .with_seed(ROTATION_SEED);
        let block = quantize_with_qjl(&config, &x, qjl_seed).unwrap();
        let est = estimate_inner_product_single(&y, &block, &config, qjl_seed).unwrap() as f64;

        distortion_sum += (true_ip - est).powi(2);
    }

    let empirical_distortion = distortion_sum / STAT_SAMPLES as f64;

    eprintln!(
        "Theorem 2 distortion (b=4, d={DIM}): empirical={empirical_distortion:.6}, \
         paper_approx={approximate_value:.6}, general_bound={general_bound:.6}"
    );

    let test_bound = general_bound * 2.0;
    assert!(
        empirical_distortion < test_bound,
        "Paper Theorem 2 distortion bound violated (b=4): \
         empirical={empirical_distortion:.6} > 2×bound={test_bound:.6}"
    );
}

// ---------------------------------------------------------------------------
// PolarQuant-only bias (Section 3.2 of paper)
//
// Paper: "for large enough d, E[⟨y, Q_mse^{-1}(Q_mse(x))⟩] = 2/π · ⟨y, x⟩"
// This is for b=1 specifically. For b=2 (TQ3's polar part), bias diminishes
// but is still nonzero.
// ---------------------------------------------------------------------------

/// Paper Section 3.2: PolarQuant without QJL has multiplicative bias.
///
/// The 2/π bias is MULTIPLICATIVE: E[⟨y, x̃_mse⟩] = α·⟨y, x⟩ where α < 1.
/// For random unit vectors, E[⟨y,x⟩] = 0, so the additive bias is zero.
/// We detect the multiplicative bias by measuring the SLOPE of
/// polar_estimate vs true_ip (should be < 1.0 for polar, = 1.0 for QJL).
///
/// Equivalently: E[polar_ip · true_ip] / E[true_ip²] < 1.0
#[test]
fn polar_only_has_multiplicative_bias_qjl_fixes_it() {
    let total_bits: u8 = 3;
    let polar_bits = total_bits - 1;

    let mut polar_xy_sum = 0.0_f64; // Σ polar_ip × true_ip
    let mut qjl_xy_sum = 0.0_f64; // Σ qjl_ip × true_ip
    let mut true_sq_sum = 0.0_f64; // Σ true_ip²

    for i in 0..STAT_SAMPLES {
        let qjl_seed = 55555_u64.wrapping_add(i as u64);

        let x = random_unit_vec(DIM, i as u64 * 61 + 7000);
        let y = random_unit_vec(DIM, i as u64 * 67 + 8000);
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

// ---------------------------------------------------------------------------
// Algorithm 2 structural verification
//
// Verify that the inner product estimate matches the paper's formula:
//   ⟨y, x̃⟩ = ⟨y, x̃_mse⟩ + √(π/2)/√d · γ · ⟨S·y, qjl⟩
// ---------------------------------------------------------------------------

/// Verify estimate_inner_product matches Algorithm 2's formula manually.
#[test]
fn algorithm2_formula_matches_implementation() {
    use turboquant::precompute_query_projections;

    let total_bits: u8 = 3;
    let polar_bits = total_bits - 1;
    let qjl_seed: u64 = 42424;

    let x = random_unit_vec(DIM, 11111);
    let y = random_unit_vec(DIM, 22222);

    let config = TurboQuantConfig::new(total_bits, DIM)
        .unwrap()
        .with_seed(ROTATION_SEED);
    let polar_config = TurboQuantConfig::new(polar_bits, DIM)
        .unwrap()
        .with_seed(ROTATION_SEED);

    // --- turboquant-rs result ---
    let block = quantize_with_qjl(&config, &x, qjl_seed).unwrap();
    let crate_estimate = estimate_inner_product_single(&y, &block, &config, qjl_seed).unwrap();

    // --- Manual Algorithm 2 computation ---
    // Step 1: x̃_mse = DeQuantmse(idx) with (b-1) bits
    let x_mse = dequantize_vec(&polar_config, block.polar_block()).unwrap();
    // Step 2: base = ⟨y, x̃_mse⟩
    let base = dot_product(&y, &x_mse);
    // Step 3: γ = ‖r‖₂
    let gamma = block.residual_norm().to_f32();
    // Step 4: c = √(π/2)/√d · γ
    let c = gamma * (SQRT_PI_OVER_2 as f32) / (DIM as f32).sqrt();
    // Step 5: correction = Σ_j (S·y)_j · qjl_j
    let s_y = precompute_query_projections(&y, DIM, qjl_seed);
    let signs = block.qjl_signs();
    let correction: f32 = s_y
        .iter()
        .enumerate()
        .take(DIM)
        .map(|(j, &sy_j)| sy_j * sign_bit(signs, j))
        .sum();
    // Step 6: full estimate = base + c · correction
    let manual_estimate = base + c * correction;

    let diff = (crate_estimate - manual_estimate).abs();
    assert!(
        diff < 1e-5,
        "Algorithm 2 formula mismatch: crate={crate_estimate:.6}, \
         manual={manual_estimate:.6}, diff={diff:.2e}. \
         turboquant-rs may not implement Algorithm 2 correctly."
    );

    // Also verify scaling constant
    let c_from_crate = qjl_scaling_constant(gamma, DIM);
    let c_diff = (c - c_from_crate).abs();
    assert!(
        c_diff < 1e-7,
        "Scaling constant mismatch: manual={c:.6}, crate={c_from_crate:.6}"
    );
}

// ---------------------------------------------------------------------------
// WHT self-inverse property
// ---------------------------------------------------------------------------

/// Paper Section 3.1: normalized WHT is self-inverse: WHT(WHT(x)) = x.
#[test]
fn wht_is_self_inverse() {
    for dim in [64, 128, 256] {
        let original = pseudo_random_vec(dim, 31415);

        let mut transformed = original.clone();
        wht_inplace(&mut transformed);
        wht_inplace(&mut transformed);

        let max_diff: f32 = original
            .iter()
            .zip(transformed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);

        assert!(
            max_diff < 1e-5,
            "WHT not self-inverse at dim={dim}: max_diff={max_diff:.2e}"
        );
    }
}

// ---------------------------------------------------------------------------
// Compression ratio verification
// ---------------------------------------------------------------------------

/// Paper Abstract: "compressing quantized vectors by at least a factor of 4.5×"
#[test]
fn compression_ratio_matches_paper() {
    let dim: usize = 128;
    let polar_bits: u8 = 2; // TQ3 polar part

    let polar_index_bytes = dim * (polar_bits as usize) / 8;
    let scale_bytes: usize = 2; // f16
    let qjl_sign_bytes = dim / 8; // 1 bit per dim
    let residual_norm_bytes: usize = 2; // f16

    let total_tq3_bytes = polar_index_bytes + scale_bytes + qjl_sign_bytes + residual_norm_bytes;
    let fp16_bytes = dim * 2;
    let compression = fp16_bytes as f64 / total_tq3_bytes as f64;

    assert_eq!(polar_index_bytes, 32, "2-bit x 128 = 32 bytes");
    assert_eq!(qjl_sign_bytes, 16, "1-bit x 128 = 16 bytes");
    assert_eq!(total_tq3_bytes, 52, "Total TQ3: 32 + 2 + 16 + 2 = 52 bytes");
    assert_eq!(fp16_bytes, 256, "FP16: 128 x 2 = 256 bytes");

    let min_compression = 4.5;
    assert!(
        compression >= min_compression,
        "Compression {compression:.2}x below paper's {min_compression}x claim"
    );
}

// ---------------------------------------------------------------------------
// Residual norm consistency (Algorithm 2, line 6-8)
// ---------------------------------------------------------------------------

/// Residual norm stored in QjlBlock must equal L2(x - dequant(quant(x))).
#[test]
fn residual_norm_equals_quantization_error() {
    let total_bits: u8 = 3;
    let polar_bits = total_bits - 1;

    for i in 0..20 {
        let x = random_unit_vec(DIM, i * 71 + 100);
        let config = TurboQuantConfig::new(total_bits, DIM)
            .unwrap()
            .with_seed(ROTATION_SEED);
        let polar_config = TurboQuantConfig::new(polar_bits, DIM)
            .unwrap()
            .with_seed(ROTATION_SEED);

        let qjl_seed = 13579_u64.wrapping_add(i);
        let block = quantize_with_qjl(&config, &x, qjl_seed).unwrap();

        let x_mse = dequantize_vec(&polar_config, block.polar_block()).unwrap();
        let residual_norm_manual: f32 = x
            .iter()
            .zip(x_mse.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        let residual_norm_stored = block.residual_norm().to_f32();

        let rel_diff = if residual_norm_manual > 1e-8 {
            (residual_norm_stored - residual_norm_manual).abs() / residual_norm_manual
        } else {
            (residual_norm_stored - residual_norm_manual).abs()
        };

        assert!(
            rel_diff < 0.02,
            "Residual norm mismatch at sample {i}: \
             stored={residual_norm_stored:.6}, manual={residual_norm_manual:.6}, \
             rel_diff={rel_diff:.4}"
        );
    }
}
