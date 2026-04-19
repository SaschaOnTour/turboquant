//! Paper verification: Algorithm 2 formula, residual norm, WHT, compression ratio
//!
//! Verifies the TurboQuant paper (Zandieh et al., ICLR 2026) against
//! the implementation. Extracted from the former
//! `paper_verification_tests.rs`.

use turboquant::packed::TurboQuantConfig;
use turboquant::qjl::{
    dot_product, estimate_inner_product_single, qjl_scaling_constant, quantize_with_qjl, sign_bit,
};
use turboquant::quantize::dequantize_vec;
use turboquant::rotation::wht_inplace;
use turboquant::test_utils::{random_unit_vec, splitmix_random_vec};

/// Test dimension (power of two for WHT).
const DIM: usize = 128;
/// Rotation seed.
const ROTATION_SEED: u64 = 42;

// Paper Algorithm 2: QJL dequant scaling factor √(π/2).
const SQRT_PI_OVER_2: f64 = 1.253_314_137_315_500_3;
const ALGORITHM2_SEED: u64 = 42_424;
const RESIDUAL_SEED: u64 = 13_579;
const SEED_PRIME_RESIDUAL: u64 = 71;
const PAPER_COMPRESSION_RATIO: f64 = 4.5;

/// Verify estimate_inner_product matches Algorithm 2's formula manually.
#[test]
fn algorithm2_formula_matches_implementation() {
    use turboquant::precompute_query_projections;

    let total_bits: u8 = 3;
    let polar_bits = total_bits - 1;
    let qjl_seed: u64 = ALGORITHM2_SEED;

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
    let x_mse = dequantize_vec(&polar_config, &block.polar_block).unwrap();
    // Step 2: base = ⟨y, x̃_mse⟩
    let base = dot_product(&y, &x_mse);
    // Step 3: γ = ‖r‖₂
    let gamma = block.residual_norm.to_f32();
    // Step 4: c = √(π/2)/√d · γ
    let c = gamma * (SQRT_PI_OVER_2 as f32) / (DIM as f32).sqrt();
    // Step 5: correction = Σ_j (S·y)_j · qjl_j
    let s_y = precompute_query_projections(&y, DIM, qjl_seed);
    let signs = &block.qjl_signs;
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

/// Paper Section 3.1: normalized WHT is self-inverse: WHT(WHT(x)) = x.
#[test]
fn wht_is_self_inverse() {
    for dim in [64, 128, 256] {
        let original = splitmix_random_vec(dim, 31415);

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

/// Paper Abstract: "compressing quantized vectors by at least a factor of 4.5×"
// qual:allow(no_sut) — verifies the paper's byte-count formula, not a function; values are compared as pure arithmetic (no SUT call to instrument)
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

    let min_compression = PAPER_COMPRESSION_RATIO;
    assert!(
        compression >= min_compression,
        "Compression {compression:.2}x below paper's {min_compression}x claim"
    );
}

/// Residual norm stored in QjlBlock must equal L2(x - dequant(quant(x))).
#[test]
fn residual_norm_equals_quantization_error() {
    let total_bits: u8 = 3;
    let polar_bits = total_bits - 1;

    for i in 0..20 {
        let x = random_unit_vec(DIM, i * SEED_PRIME_RESIDUAL + 100);
        let config = TurboQuantConfig::new(total_bits, DIM)
            .unwrap()
            .with_seed(ROTATION_SEED);
        let polar_config = TurboQuantConfig::new(polar_bits, DIM)
            .unwrap()
            .with_seed(ROTATION_SEED);

        let qjl_seed = RESIDUAL_SEED.wrapping_add(i);
        let block = quantize_with_qjl(&config, &x, qjl_seed).unwrap();

        let x_mse = dequantize_vec(&polar_config, &block.polar_block).unwrap();
        let residual_norm_manual: f32 = x
            .iter()
            .zip(x_mse.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        let residual_norm_stored = block.residual_norm.to_f32();

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
