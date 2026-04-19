//! Rotation tests extracted from the former `roundtrip_tests.rs`.

// qual:allow(srp) — cohesive test module: rotation / WHT / sign-pattern tests
use approx::assert_abs_diff_eq;
use turboquant::rotation::{generate_sign_pattern, rotate, wht_inplace, RotationOrder};
use turboquant::test_utils::pseudo_random_vec;

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

/// Computes the L2 norm of a slice.
fn l2_norm(data: &[f32]) -> f32 {
    data.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// -----------------------------------------------------------------------
// WHT norm preservation
// -----------------------------------------------------------------------

#[test]
fn wht_preserves_norm_dim64() {
    wht_preserves_norm(64);
}

#[test]
fn wht_preserves_norm_dim128() {
    wht_preserves_norm(128);
}

#[test]
fn wht_preserves_norm_dim256() {
    wht_preserves_norm(256);
}

fn wht_preserves_norm(dim: usize) {
    let mut data = pseudo_random_vec(dim, 12345);
    let norm_before = l2_norm(&data);

    wht_inplace(&mut data);
    let norm_after = l2_norm(&data);

    assert_abs_diff_eq!(norm_before, norm_after, epsilon = 1e-3);
}

// -----------------------------------------------------------------------
// WHT self-inversity
// -----------------------------------------------------------------------

#[test]
fn wht_is_self_inverse_dim64() {
    wht_is_self_inverse(64);
}

#[test]
fn wht_is_self_inverse_dim128() {
    wht_is_self_inverse(128);
}

#[test]
fn wht_is_self_inverse_dim256() {
    wht_is_self_inverse(256);
}

fn wht_is_self_inverse(dim: usize) {
    let original = pseudo_random_vec(dim, 54321);
    let mut data = original.clone();

    wht_inplace(&mut data);
    wht_inplace(&mut data);

    for (a, b) in original.iter().zip(data.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-4);
    }
}

// -----------------------------------------------------------------------
// validate_rotation_inputs rejects non-power-of-two (via rotate)
// -----------------------------------------------------------------------

#[test]
fn validate_rotation_rejects_non_power_of_two() {
    let mut data = vec![1.0; 3];
    let signs = vec![1.0; 3];
    assert!(rotate(&mut data, &signs, RotationOrder::Forward).is_err());
}

#[test]
fn validate_rotation_accepts_power_of_two() {
    let mut data = vec![1.0; 8];
    let signs = generate_sign_pattern(8, 42);
    assert!(rotate(&mut data, &signs, RotationOrder::Forward).is_ok());
}

// -----------------------------------------------------------------------
// Sign-pattern determinism
// -----------------------------------------------------------------------

#[test]
fn same_seed_produces_same_sign_pattern() {
    let a = generate_sign_pattern(256, 42);
    let b = generate_sign_pattern(256, 42);
    assert_eq!(a, b);
}

#[test]
fn different_seeds_produce_different_sign_patterns() {
    let a = generate_sign_pattern(256, 1);
    let b = generate_sign_pattern(256, 2);
    // They could theoretically match, but with 256 elements it is
    // astronomically unlikely.
    assert_ne!(a, b);
}

#[test]
fn sign_pattern_contains_only_plus_minus_one() {
    let pattern = generate_sign_pattern(512, 77);
    for &v in &pattern {
        assert!(v == 1.0 || v == -1.0, "unexpected value: {v}");
    }
}

// -----------------------------------------------------------------------
// Full rotation roundtrip
// -----------------------------------------------------------------------

#[test]
fn rotation_roundtrip_dim64() {
    rotation_roundtrip(64, 100);
}

#[test]
fn rotation_roundtrip_dim128() {
    rotation_roundtrip(128, 200);
}

#[test]
fn rotation_roundtrip_dim256() {
    rotation_roundtrip(256, 300);
}

fn rotation_roundtrip(dim: usize, seed: u64) {
    let original = pseudo_random_vec(dim, seed);
    let sign_pattern = generate_sign_pattern(dim, seed);

    let mut data = original.clone();
    rotate(&mut data, &sign_pattern, RotationOrder::Forward).expect("rotate should succeed");
    rotate(&mut data, &sign_pattern, RotationOrder::Inverse)
        .expect("inverse_rotate should succeed");

    for (a, b) in original.iter().zip(data.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-4);
    }
}

// -----------------------------------------------------------------------
// Rotation preserves norm
// -----------------------------------------------------------------------

#[test]
fn rotation_preserves_norm() {
    let dim = 128;
    let seed = 55;
    let sign_pattern = generate_sign_pattern(dim, seed);
    let mut data = pseudo_random_vec(dim, seed);
    let norm_before = l2_norm(&data);

    rotate(&mut data, &sign_pattern, RotationOrder::Forward).expect("rotate should succeed");
    let norm_after = l2_norm(&data);

    assert_abs_diff_eq!(norm_before, norm_after, epsilon = 1e-3);
}

// -----------------------------------------------------------------------
// Distribution test: rotated coordinates should have mean ~ 0
// -----------------------------------------------------------------------

#[test]
fn rotated_coordinates_have_zero_mean() {
    /// Fixed sign-pattern seed for the rotated-coordinates zero-mean test.
    const SIGN_PATTERN_SEED: u64 = 999;
    let dim = 256;
    let num_samples = 50;
    let mut total_mean = 0.0_f64;

    for sample_seed in 0..num_samples {
        let sign_pattern = generate_sign_pattern(dim, SIGN_PATTERN_SEED);
        let mut data = pseudo_random_vec(dim, 1000 + sample_seed);

        // Normalize to unit vector
        let norm = l2_norm(&data);
        if norm > 0.0 {
            for v in data.iter_mut() {
                *v /= norm;
            }
        }

        rotate(&mut data, &sign_pattern, RotationOrder::Forward).expect("rotate should succeed");

        let mean: f64 = data.iter().map(|&x| x as f64).sum::<f64>() / dim as f64;
        total_mean += mean;
    }

    let avg_mean = total_mean / num_samples as f64;
    assert!(
        avg_mean.abs() < 0.05,
        "average mean across samples should be near zero, got {avg_mean}"
    );
}

// -----------------------------------------------------------------------
// Distribution test: variance of rotated unit vectors
// -----------------------------------------------------------------------

#[test]
fn rotated_unit_vector_has_expected_variance() {
    let dim = 256;
    let sign_pattern = generate_sign_pattern(dim, 7777);
    let mut data = pseudo_random_vec(dim, 8888);

    // Normalize to unit vector
    let norm = l2_norm(&data);
    for v in data.iter_mut() {
        *v /= norm;
    }

    rotate(&mut data, &sign_pattern, RotationOrder::Forward).expect("rotate should succeed");

    // For a rotated unit vector, each coordinate has variance 1/d
    let expected_variance = 1.0_f64 / dim as f64;
    let mean: f64 = data.iter().map(|&x| x as f64).sum::<f64>() / dim as f64;
    let variance: f64 = data
        .iter()
        .map(|&x| {
            let diff = x as f64 - mean;
            diff * diff
        })
        .sum::<f64>()
        / dim as f64;

    // The variance should be close to 1/d = 0.00390625 for d=256.
    // Allow generous tolerance since this is a single sample.
    assert_abs_diff_eq!(variance, expected_variance, epsilon = 0.005);
}

// -----------------------------------------------------------------------
// Error cases
// -----------------------------------------------------------------------

#[test]
fn rotate_rejects_non_power_of_two() {
    let mut data = vec![1.0; 5];
    let signs = vec![1.0; 5];
    assert!(rotate(&mut data, &signs, RotationOrder::Forward).is_err());
}

#[test]
fn rotate_rejects_dimension_mismatch() {
    let mut data = vec![1.0; 8];
    let signs = vec![1.0; 4];
    assert!(rotate(&mut data, &signs, RotationOrder::Forward).is_err());
}

#[test]
fn inverse_rotate_rejects_non_power_of_two() {
    let mut data = vec![1.0; 6];
    let signs = vec![1.0; 6];
    assert!(rotate(&mut data, &signs, RotationOrder::Inverse).is_err());
}

#[test]
fn inverse_rotate_rejects_dimension_mismatch() {
    let mut data = vec![1.0; 16];
    let signs = vec![1.0; 8];
    assert!(rotate(&mut data, &signs, RotationOrder::Inverse).is_err());
}
