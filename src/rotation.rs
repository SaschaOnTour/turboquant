//! Walsh-Hadamard Transform and random rotation.
//!
//! The rotation transforms arbitrary input vectors into a known distribution
//! using a randomized orthogonal transform, making quantization "data-oblivious".

use crate::error::{require, values_match, Result, TurboQuantError};
use crate::packed::is_valid_dim;

// ---------------------------------------------------------------------------
// Walsh-Hadamard Transform (pure Operation)
// ---------------------------------------------------------------------------

/// Fast Walsh-Hadamard Transform, in-place, O(d log d).
///
/// The transform is normalized by `1 / sqrt(n)` so that applying it twice
/// returns the original vector (self-inverse property).
///
/// # Panics
///
/// Does **not** panic; callers should use [`is_valid_dim`](crate::packed::is_valid_dim)
/// beforehand or call [`validate_rotation_inputs`].
pub fn wht_inplace(data: &mut [f32]) {
    let n = data.len();
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                let x = data[j];
                let y = data[j + h];
                data[j] = x + y;
                data[j + h] = x - y;
            }
        }
        h *= 2;
    }
    let norm = 1.0 / (n as f32).sqrt();
    for v in data.iter_mut() {
        *v *= norm;
    }
}

// ---------------------------------------------------------------------------
// Sign-pattern generation (pure Operation)
// ---------------------------------------------------------------------------

/// Golden-ratio constant used for deterministic hashing.
const GOLDEN_RATIO: u64 = 0x9E37_79B9_7F4A_7C15;

/// Hashes `(seed, index)` deterministically using a golden-ratio multiply-shift.
fn golden_ratio_hash(seed: u64, index: usize) -> u64 {
    let combined = seed.wrapping_add(index as u64);
    combined.wrapping_mul(GOLDEN_RATIO)
}

/// Generates a deterministic sign pattern of `+1.0` / `-1.0` values.
///
/// The same `(dim, seed)` pair always produces the identical pattern.
/// Each element is `+1.0` when the hash has an even least-significant bit,
/// and `-1.0` otherwise.
pub fn generate_sign_pattern(dim: usize, seed: u64) -> Vec<f32> {
    (0..dim)
        .map(|i| {
            if golden_ratio_hash(seed, i) & 1 == 0 {
                1.0_f32
            } else {
                -1.0_f32
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Element-wise sign flip (pure Operation)
// ---------------------------------------------------------------------------

/// Multiplies each element of `data` by the corresponding element in `signs`.
///
/// # Precondition
///
/// `data.len() == signs.len()` -- caller is responsible for ensuring this.
fn apply_sign_flip(data: &mut [f32], signs: &[f32]) {
    for (v, &s) in data.iter_mut().zip(signs.iter()) {
        *v *= s;
    }
}

// ---------------------------------------------------------------------------
// Argument validation helpers (pure Operation)
// ---------------------------------------------------------------------------

/// Validates rotation inputs and returns `Ok(())` or the appropriate error.
///
/// Pure Integration: only calls `require`, `is_valid_dim` (from `packed`),
/// and `values_match` (from `error`).
fn validate_rotation_inputs(data_len: usize, sign_len: usize) -> Result<()> {
    require(
        is_valid_dim(data_len),
        TurboQuantError::InvalidDimension(data_len),
    )?;
    require(
        values_match(data_len, sign_len),
        TurboQuantError::DimensionMismatch {
            expected: data_len,
            actual: sign_len,
        },
    )
}

// ---------------------------------------------------------------------------
// Rotation order (used to DRY forward / inverse rotation)
// ---------------------------------------------------------------------------

/// Selects the order of operations in the rotation transform.
pub enum RotationOrder {
    /// Forward: sign flip first, then WHT.
    Forward,
    /// Inverse: WHT first, then sign flip.
    Inverse,
}

/// Applies a rotation transform to `data` using the given `sign_pattern`.
///
/// - [`RotationOrder::Forward`]: element-wise sign flip, then WHT.
/// - [`RotationOrder::Inverse`]: WHT, then element-wise sign flip.
///
/// The transform is orthogonal and preserves the L2 norm of the input.
/// Forward followed by Inverse (or vice versa) recovers the original vector.
///
/// # Errors
///
/// Returns [`TurboQuantError::InvalidDimension`] if `data.len()` is not a
/// power of two, or [`TurboQuantError::DimensionMismatch`] if `data` and
/// `sign_pattern` differ in length.
///
/// Integration: validates via `validate_rotation_inputs`, then applies the
/// two steps in the order determined by `order`.
pub fn rotate(data: &mut [f32], sign_pattern: &[f32], order: RotationOrder) -> Result<()> {
    validate_rotation_inputs(data.len(), sign_pattern.len())?;
    match order {
        RotationOrder::Forward => {
            apply_sign_flip(data, sign_pattern);
            wht_inplace(data);
        }
        RotationOrder::Inverse => {
            wht_inplace(data);
            apply_sign_flip(data, sign_pattern);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small dimension used for rotation round-trip and acceptance tests.
    const TEST_SMALL_DIM: usize = 4;
    /// Seed used for deterministic sign-pattern generation in rotation tests.
    const TEST_SEED: u64 = 42;
    /// Dimension used for sign-pattern generation tests (must be a power of two).
    const TEST_SIGN_PATTERN_DIM: usize = 128;
    /// Seed used for deterministic sign-pattern generation tests.
    const TEST_SIGN_PATTERN_SEED: u64 = 99;
    /// Index used in golden_ratio_hash determinism test.
    const TEST_HASH_INDEX: usize = 7;

    // -- is_valid_dim --------------------------------------------------

    #[test]
    fn is_valid_dim_accepts_powers_of_two() {
        assert!(is_valid_dim(1));
        assert!(is_valid_dim(2));
        assert!(is_valid_dim(64));
        assert!(is_valid_dim(256));
    }

    #[test]
    fn is_valid_dim_rejects_invalid() {
        assert!(!is_valid_dim(0));
        assert!(!is_valid_dim(3));
        assert!(!is_valid_dim(100));
    }

    // -- golden_ratio_hash ---------------------------------------------------

    #[test]
    fn golden_ratio_hash_is_deterministic() {
        let a = golden_ratio_hash(TEST_SEED, TEST_HASH_INDEX);
        let b = golden_ratio_hash(TEST_SEED, TEST_HASH_INDEX);
        assert_eq!(a, b);
    }

    // -- sign_pattern_elements_are_plus_or_minus_one -------------------------

    #[test]
    fn sign_pattern_elements_are_plus_or_minus_one() {
        let pattern = generate_sign_pattern(TEST_SIGN_PATTERN_DIM, TEST_SIGN_PATTERN_SEED);
        assert_sign_values_valid(&pattern);
    }

    /// Pure assertion helper: checks every element is +1.0 or -1.0.
    ///
    /// Separated from generation so the test is not mixing calls with loop logic.
    fn assert_sign_values_valid(pattern: &[f32]) {
        for &v in pattern {
            assert!(v == 1.0 || v == -1.0);
        }
    }

    // -- validate_rotation_args (via rotate / inverse_rotate) ----------------

    #[test]
    fn rotate_accepts_matching_pow2_dims() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let signs = generate_sign_pattern(TEST_SMALL_DIM, TEST_SEED);
        assert!(rotate(&mut data, &signs, RotationOrder::Forward).is_ok());
    }

    #[test]
    fn rotate_rejects_non_pow2() {
        let mut data = vec![1.0, 2.0, 3.0];
        let signs = vec![1.0, -1.0, 1.0];
        assert!(rotate(&mut data, &signs, RotationOrder::Forward).is_err());
    }

    #[test]
    fn rotate_rejects_mismatched_lengths() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let signs = vec![1.0, -1.0];
        assert!(rotate(&mut data, &signs, RotationOrder::Forward).is_err());
    }

    // -- wht_inplace ---------------------------------------------------------

    #[test]
    fn wht_inplace_known_vector() {
        // WHT of [1, 1, 1, 1] normalized by 1/sqrt(4) = 0.5
        // -> [2, 0, 0, 0] * 0.5 = [1, 0, 0, 0]
        let mut data = vec![1.0, 1.0, 1.0, 1.0];
        wht_inplace(&mut data);
        assert!((data[0] - 2.0).abs() < 1e-6);
        assert!((data[1] - 0.0).abs() < 1e-6);
        assert!((data[2] - 0.0).abs() < 1e-6);
        assert!((data[3] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn wht_inplace_self_inverse() {
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let mut data = original.clone();
        wht_inplace(&mut data);
        wht_inplace(&mut data);
        for (a, b) in data.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    // -- apply_sign_flip -----------------------------------------------------

    #[test]
    fn apply_sign_flip_basic() {
        let mut data = vec![2.0, -3.0, 4.0, -5.0];
        let signs = vec![1.0, -1.0, -1.0, 1.0];
        apply_sign_flip(&mut data, &signs);
        assert_eq!(data, vec![2.0, 3.0, -4.0, -5.0]);
    }

    // -- values_match (from error.rs) -----------------------------------------

    #[test]
    fn values_match_equal() {
        assert!(values_match(4, 4));
        assert!(values_match(128, 128));
    }

    #[test]
    fn values_match_unequal() {
        assert!(!values_match(4, 8));
        assert!(!values_match(0, 1));
    }

    // -- roundtrip rotation --------------------------------------------------

    #[test]
    fn rotate_inverse_rotate_roundtrip() {
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let mut data = original.clone();
        let signs = generate_sign_pattern(TEST_SMALL_DIM, TEST_SEED);
        rotate(&mut data, &signs, RotationOrder::Forward).unwrap();
        rotate(&mut data, &signs, RotationOrder::Inverse).unwrap();
        for (a, b) in data.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}
