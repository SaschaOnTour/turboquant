//! Quantize roundtrip tests extracted from the former `roundtrip_tests.rs`.

// qual:allow(srp) — cohesive integration-test module
use approx::assert_abs_diff_eq;
use turboquant::packed::TurboQuantConfig;
use turboquant::quantize::{dequantize_rotated, dequantize_vec, l2_norm, quantize_vec};
use turboquant::test_utils::pseudo_random_vec;

// -----------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------

/// Default seed for rotation.
const TEST_SEED: u64 = 42;
/// Tolerance for norm comparisons after quantization roundtrip.
/// 3-bit quantization introduces ~18% relative error on average (sqrt(MSE=0.034)),
/// so the norm can deviate significantly.  f16 rounding adds further noise.
const NORM_EPSILON: f32 = 0.35;
/// Tolerance for near-zero checks.
const ZERO_EPSILON: f32 = 0.1;

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

/// Computes the squared error between two vectors.
fn squared_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum()
}

// -----------------------------------------------------------------------
// Roundtrip: dequantize(quantize(x)) is close to x
// -----------------------------------------------------------------------

#[test]
fn roundtrip_tq3_dim64() {
    roundtrip_check(3, 64, 1000);
}

#[test]
fn roundtrip_tq3_dim128() {
    roundtrip_check(3, 128, 2000);
}

#[test]
fn roundtrip_tq3_dim256() {
    roundtrip_check(3, 256, 3000);
}

#[test]
fn roundtrip_tq4_dim64() {
    roundtrip_check(4, 64, 4000);
}

#[test]
fn roundtrip_tq4_dim128() {
    roundtrip_check(4, 128, 5000);
}

#[test]
fn roundtrip_tq4_dim256() {
    roundtrip_check(4, 256, 6000);
}

fn roundtrip_check(bits: u8, dim: usize, seed: u64) {
    let config = TurboQuantConfig::new(bits, dim)
        .unwrap()
        .with_seed(TEST_SEED);
    let data = pseudo_random_vec(dim, seed);
    let block = quantize_vec(&config, &data).unwrap();
    let recovered = dequantize_vec(&config, &block).unwrap();

    let orig_norm_sq = data.iter().map(|&x| x * x).sum::<f32>();
    let err_sq = squared_error(&data, &recovered);
    let relative_mse = err_sq / orig_norm_sq;

    // Single-vector relative MSE can be much higher than the aggregate
    // mean (0.034 for TQ3, 0.009 for TQ4, ~0.10 for TQ2). The proper
    // quality gate is mse_validation.rs which checks over 10,000 vectors.
    let threshold = match bits {
        2 => 1.5,
        3 => 1.0,
        _ => 0.5,
    };
    assert!(
        relative_mse < threshold,
        "bits={bits}, dim={dim}: relative MSE {relative_mse} exceeds {threshold}"
    );
}

// -----------------------------------------------------------------------
// Null vector: quantize([0,...,0]) doesn't panic, dequantize gives zeros
// -----------------------------------------------------------------------

#[test]
fn null_vector_tq3() {
    null_vector_check(3, 128);
}

#[test]
fn null_vector_tq4() {
    null_vector_check(4, 128);
}

fn null_vector_check(bits: u8, dim: usize) {
    let config = TurboQuantConfig::new(bits, dim)
        .unwrap()
        .with_seed(TEST_SEED);
    let data = vec![0.0_f32; dim];
    let block = quantize_vec(&config, &data).unwrap();
    let recovered = dequantize_vec(&config, &block).unwrap();
    let norm = l2_norm(&recovered);
    assert!(
        norm < ZERO_EPSILON,
        "null vector roundtrip should give near-zero, got norm={norm}"
    );
}

// -----------------------------------------------------------------------
// Unit vector: quantize(e1) works correctly
// -----------------------------------------------------------------------

#[test]
fn unit_vector_tq3() {
    unit_vector_check(3, 128);
}

#[test]
fn unit_vector_tq4() {
    unit_vector_check(4, 128);
}

fn unit_vector_check(bits: u8, dim: usize) {
    let config = TurboQuantConfig::new(bits, dim)
        .unwrap()
        .with_seed(TEST_SEED);
    let mut data = vec![0.0_f32; dim];
    data[0] = 1.0;
    let block = quantize_vec(&config, &data).unwrap();
    let recovered = dequantize_vec(&config, &block).unwrap();

    // The recovered vector should have a non-zero norm in the right
    // ballpark.  Exact norm preservation is not guaranteed by scalar
    // quantization.
    let rec_norm = l2_norm(&recovered);
    assert!(rec_norm > 0.3, "recovered norm too small: {rec_norm}");
    assert!(rec_norm < 2.0, "recovered norm too large: {rec_norm}");
}

// -----------------------------------------------------------------------
// Constant vector: all same value
// -----------------------------------------------------------------------

#[test]
fn constant_vector_tq3() {
    constant_vector_check(3, 128);
}

#[test]
fn constant_vector_tq4() {
    constant_vector_check(4, 128);
}

fn constant_vector_check(bits: u8, dim: usize) {
    let config = TurboQuantConfig::new(bits, dim)
        .unwrap()
        .with_seed(TEST_SEED);
    let val = 0.5_f32;
    let data = vec![val; dim];
    let block = quantize_vec(&config, &data).unwrap();
    let recovered = dequantize_vec(&config, &block).unwrap();

    // Verify the pipeline doesn't blow up on constant vectors and the
    // recovered norm is in a reasonable range.
    let orig_norm = l2_norm(&data);
    let rec_norm = l2_norm(&recovered);
    let ratio = rec_norm / orig_norm;
    // Constant vectors are adversarial for rotation-based quantization
    // (all energy concentrates in one WHT coefficient), so the ratio
    // can be quite low.
    assert!(ratio > 0.1, "recovered norm too small: ratio={ratio}");
    assert!(ratio < 3.0, "recovered norm too large: ratio={ratio}");
}

// -----------------------------------------------------------------------
// Determinism: same input + config -> identical output
// -----------------------------------------------------------------------

#[test]
fn determinism_tq3() {
    determinism_check(3, 128, 11111);
}

#[test]
fn determinism_tq4() {
    determinism_check(4, 128, 22222);
}

fn determinism_check(bits: u8, dim: usize, seed: u64) {
    let config = TurboQuantConfig::new(bits, dim)
        .unwrap()
        .with_seed(TEST_SEED);
    let data = pseudo_random_vec(dim, seed);

    let block_a = quantize_vec(&config, &data).unwrap();
    let block_b = quantize_vec(&config, &data).unwrap();

    let rec_a = dequantize_vec(&config, &block_a).unwrap();
    let rec_b = dequantize_vec(&config, &block_b).unwrap();

    assert_eq!(rec_a, rec_b, "quantization should be deterministic");
}

// -----------------------------------------------------------------------
// Different dimensions: d=64, d=128, d=256
// -----------------------------------------------------------------------

#[test]
fn different_dimensions_tq3() {
    for &dim in &[64, 128, 256] {
        let config = TurboQuantConfig::new(3, dim).unwrap().with_seed(TEST_SEED);
        let data = pseudo_random_vec(dim, dim as u64);
        let block = quantize_vec(&config, &data).unwrap();
        let recovered = dequantize_vec(&config, &block).unwrap();
        assert_eq!(recovered.len(), dim);
    }
}

#[test]
fn different_dimensions_tq4() {
    for &dim in &[64, 128, 256] {
        let config = TurboQuantConfig::new(4, dim).unwrap().with_seed(TEST_SEED);
        let data = pseudo_random_vec(dim, dim as u64 + 1000);
        let block = quantize_vec(&config, &data).unwrap();
        let recovered = dequantize_vec(&config, &block).unwrap();
        assert_eq!(recovered.len(), dim);
    }
}

// -----------------------------------------------------------------------
// dequantize_rotated: differs from full dequantize but same norm
// -----------------------------------------------------------------------

#[test]
fn dequantize_rotated_differs_but_same_norm_tq3() {
    dequantize_rotated_check(3, 128, 33333);
}

#[test]
fn dequantize_rotated_differs_but_same_norm_tq4() {
    dequantize_rotated_check(4, 128, 44444);
}

fn dequantize_rotated_check(bits: u8, dim: usize, seed: u64) {
    let config = TurboQuantConfig::new(bits, dim)
        .unwrap()
        .with_seed(TEST_SEED);
    let data = pseudo_random_vec(dim, seed);
    let block = quantize_vec(&config, &data).unwrap();

    let full = dequantize_vec(&config, &block).unwrap();
    let rotated = dequantize_rotated(&config, &block).unwrap();

    // Coordinates should differ.
    assert_ne!(full, rotated, "rotated and full dequantize should differ");

    // Norms should be approximately equal (rotation preserves norm).
    let full_norm = l2_norm(&full);
    let rotated_norm = l2_norm(&rotated);
    assert_abs_diff_eq!(full_norm, rotated_norm, epsilon = NORM_EPSILON);
}

// -----------------------------------------------------------------------
// PackedBlock: both TQ2, TQ3, and TQ4 work via quantize_vec
// -----------------------------------------------------------------------

#[test]
fn packed_block_tq2() {
    let config = TurboQuantConfig::new(2, 64).unwrap().with_seed(TEST_SEED);
    let data = pseudo_random_vec(64, 44444);
    let block = quantize_vec(&config, &data).unwrap();
    assert_eq!(block.bits, 2);
    let recovered = dequantize_vec(&config, &block).unwrap();
    assert_eq!(recovered.len(), 64);
}

#[test]
fn packed_block_tq3() {
    let config = TurboQuantConfig::new(3, 64).unwrap().with_seed(TEST_SEED);
    let data = pseudo_random_vec(64, 55555);
    let block = quantize_vec(&config, &data).unwrap();
    assert_eq!(block.bits, 3);
    let recovered = dequantize_vec(&config, &block).unwrap();
    assert_eq!(recovered.len(), 64);
}

#[test]
fn packed_block_tq4() {
    let config = TurboQuantConfig::new(4, 64).unwrap().with_seed(TEST_SEED);
    let data = pseudo_random_vec(64, 66666);
    let block = quantize_vec(&config, &data).unwrap();
    assert_eq!(block.bits, 4);
    let recovered = dequantize_vec(&config, &block).unwrap();
    assert_eq!(recovered.len(), 64);
}

// -----------------------------------------------------------------------
// 2-bit roundtrip tests
// -----------------------------------------------------------------------

#[test]
fn roundtrip_tq2_dim64() {
    roundtrip_check(2, 64, 7000);
}

#[test]
fn roundtrip_tq2_dim128() {
    roundtrip_check(2, 128, 8000);
}

#[test]
fn roundtrip_tq2_dim256() {
    roundtrip_check(2, 256, 9000);
}

#[test]
fn null_vector_tq2() {
    null_vector_check(2, 128);
}

#[test]
fn unit_vector_tq2() {
    unit_vector_check(2, 128);
}

#[test]
fn constant_vector_tq2() {
    constant_vector_check(2, 128);
}

#[test]
fn determinism_tq2() {
    determinism_check(2, 128, 33333);
}

#[test]
fn different_dimensions_tq2() {
    for &dim in &[64, 128, 256] {
        let config = TurboQuantConfig::new(2, dim).unwrap().with_seed(TEST_SEED);
        let data = pseudo_random_vec(dim, dim as u64 + 2000);
        let block = quantize_vec(&config, &data).unwrap();
        let recovered = dequantize_vec(&config, &block).unwrap();
        assert_eq!(recovered.len(), dim);
    }
}

#[test]
fn dequantize_rotated_differs_but_same_norm_tq2() {
    dequantize_rotated_check(2, 128, 55555);
}
