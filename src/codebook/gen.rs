//! Lloyd-Max codebook generation for the Beta distribution.
//!
//! This module contains the Lloyd-Max iterative algorithm and all generation
//! helpers.  It is separated from [`crate::codebook`] (which owns the
//! [`Codebook`] struct, static lookup tables, and the Beta PDF) to respect
//! the Single Responsibility Principle.

use super::{centroid_count, Codebook, SUPPORT_MAX, SUPPORT_MIN};
use crate::math::{converge, ln_gamma, simpsons_integrate, HALF};

// ---------------------------------------------------------------------------
// Constants — generation-specific
// ---------------------------------------------------------------------------

/// Maximum number of Lloyd-Max iterations before we declare convergence.
const MAX_ITERATIONS: usize = 200;

/// Convergence threshold: stop when the relative change in distortion drops
/// below this value.
const CONVERGENCE_EPS: f64 = 1e-12;

/// Number of sub-intervals used for Simpson's rule integration.
const INTEGRATION_STEPS: usize = 1024;

/// Small epsilon to guard against division by near-zero values.
const EPSILON_ZERO: f64 = 1e-30;

/// Minimum dimension for which the Beta-type PDF is well-defined.
/// For d < 3 the exponent (d-3)/2 is negative and the distribution degenerates.
const MIN_DIMENSION_FOR_PDF: usize = 3;

/// The exponent offset in the Beta-type kernel: (d - 3) / 2.
const KERNEL_EXPONENT_OFFSET: f64 = 3.0;

// ---------------------------------------------------------------------------
// Pure Operation: Beta PDF
// ---------------------------------------------------------------------------

/// Evaluate the Beta-type PDF of a rotated unit-vector coordinate.
///
/// ```text
/// f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)
/// ```
///
/// Pure Operation: all arithmetic (kernel + normalization) is computed
/// inline without calls to other project functions.
pub fn beta_pdf(x: f64, d: usize) -> f64 {
    // Guard: dimension too low.
    if d < MIN_DIMENSION_FOR_PDF {
        return 0.0;
    }
    let df = d as f64;
    let exponent = (df - KERNEL_EXPONENT_OFFSET) * HALF;
    let one_minus_x2 = 1.0 - x * x;
    if one_minus_x2 <= 0.0 {
        return 0.0;
    }
    let kernel = one_minus_x2.powf(exponent);

    // Normalization: ln(Gamma(d/2)) - 0.5*ln(pi) - ln(Gamma((d-1)/2))
    let half_df = df * HALF;
    let half_df_minus_one = (df - 1.0) * HALF;
    let half_ln_pi = HALF * core::f64::consts::PI.ln();
    let log_norm = ln_gamma(half_df) - half_ln_pi - ln_gamma(half_df_minus_one);

    log_norm.exp() * kernel
}

// ---------------------------------------------------------------------------
// Pure Operation: initialization
// ---------------------------------------------------------------------------

/// Place `k` centroids uniformly on `(SUPPORT_MIN, SUPPORT_MAX)` (excluding endpoints).
fn initialize_centroids(k: usize) -> Vec<f64> {
    let range = SUPPORT_MAX - SUPPORT_MIN; // 2.0
    (0..k)
        .map(|i| SUPPORT_MIN + (range * (i as f64 + HALF)) / k as f64)
        .collect()
}

/// Compute midpoint boundaries between adjacent centroids.
fn midpoint_boundaries(centroids: &[f64]) -> Vec<f64> {
    centroids.windows(2).map(|w| (w[0] + w[1]) * HALF).collect()
}

// ---------------------------------------------------------------------------
// Pure Operation: bin geometry
// ---------------------------------------------------------------------------

/// Determine the lower bound of the i-th bin given boundaries.
fn bin_lower_bound(i: usize, boundaries: &[f64]) -> f64 {
    if i == 0 {
        SUPPORT_MIN
    } else {
        boundaries[i - 1]
    }
}

/// Determine the upper bound of the i-th bin given boundaries and total
/// number of centroids `k`.
fn bin_upper_bound(i: usize, k: usize, boundaries: &[f64]) -> f64 {
    if i == k - 1 {
        SUPPORT_MAX
    } else {
        boundaries[i]
    }
}

// ---------------------------------------------------------------------------
// Pure Operation: convergence check & conditional selection
// ---------------------------------------------------------------------------

/// Check whether the Lloyd-Max iteration has converged by comparing the
/// relative change in distortion against [`CONVERGENCE_EPS`].
fn has_converged(prev_distortion: f64, distortion: f64) -> bool {
    (prev_distortion - distortion).abs() < CONVERGENCE_EPS * prev_distortion.abs().max(EPSILON_ZERO)
}

/// Select the conditional expectation or the interval midpoint depending
/// on whether the denominator is near zero.
///
/// Pure Operation: only arithmetic and comparison, no calls.
fn select_conditional_or_midpoint(numerator: f64, denominator: f64, a: f64, b: f64) -> f64 {
    if denominator.abs() < EPSILON_ZERO {
        (a + b) * HALF
    } else {
        numerator / denominator
    }
}

// ---------------------------------------------------------------------------
// Pure Integration: numerical integration wrappers
// ---------------------------------------------------------------------------

/// Simpson's rule numerical integration of `f` over `[a, b]`, using the
/// module-level [`INTEGRATION_STEPS`] constant.
///
/// Pure Integration: delegates to [`crate::math::simpsons_integrate`].
fn integrate<F: Fn(f64) -> f64>(f: F, a: f64, b: f64) -> f64 {
    simpsons_integrate(f, a, b, INTEGRATION_STEPS)
}

/// Compute `integral_a^b f(x) dx` where `f(x) = beta_pdf(x, d)`.
///
/// Pure Integration: delegates to `integrate` and `beta_pdf`.
fn integrate_pdf(a: f64, b: f64, d: usize) -> f64 {
    integrate(|x| beta_pdf(x, d), a, b)
}

/// Compute `integral_a^b x * f(x) dx` where `f(x) = beta_pdf(x, d)`.
///
/// Pure Integration: delegates to `integrate` and `beta_pdf`.
fn integrate_x_pdf(a: f64, b: f64, d: usize) -> f64 {
    integrate(|x| x * beta_pdf(x, d), a, b)
}

/// Conditional expectation `E[X | X in [a, b]]` under the Beta-type PDF.
///
/// Pure Integration: delegates computation to `integrate_pdf`,
/// `integrate_x_pdf`, and `select_conditional_or_midpoint`.
fn conditional_expectation(a: f64, b: f64, d: usize) -> f64 {
    let denom = integrate_pdf(a, b, d);
    let numer = integrate_x_pdf(a, b, d);
    select_conditional_or_midpoint(numer, denom, a, b)
}

// ---------------------------------------------------------------------------
// Pure Integration: distortion computation
// ---------------------------------------------------------------------------

/// Compute the MSE-distortion contribution of a single bin `[lo, hi]` with
/// centroid `c` under the Beta PDF for dimension `d`.
///
/// Pure Integration: delegates to `integrate` and `beta_pdf`.
fn bin_distortion(lo: f64, hi: f64, c: f64, d: usize) -> f64 {
    integrate(|x| (x - c).powi(2) * beta_pdf(x, d), lo, hi)
}

/// Compute the MSE distortion of the current codebook under the Beta PDF.
///
/// Pure Integration: delegates bin bounds to `bin_lower_bound`/`bin_upper_bound`
/// and per-bin distortion to `bin_distortion`.  Uses an iterator chain instead
/// of explicit loop logic.
fn compute_distortion(centroids: &[f64], boundaries: &[f64], d: usize) -> f64 {
    let k = centroids.len();
    centroids
        .iter()
        .enumerate()
        .map(|(i, &centroid)| {
            let lo = bin_lower_bound(i, boundaries);
            let hi = bin_upper_bound(i, k, boundaries);
            bin_distortion(lo, hi, centroid, d)
        })
        .sum()
}

// ---------------------------------------------------------------------------
// Pure Integration: centroid update
// ---------------------------------------------------------------------------

/// Compute updated centroids for one Lloyd-Max iteration.
///
/// Pure Integration: delegates bin bounds to `bin_lower_bound`/`bin_upper_bound`
/// and centroid updates to `conditional_expectation`.  Uses an iterator chain
/// instead of explicit loop logic.
fn update_centroids(centroids_len: usize, boundaries: &[f64], d: usize) -> Vec<f64> {
    (0..centroids_len)
        .map(|i| {
            let lo = bin_lower_bound(i, boundaries);
            let hi = bin_upper_bound(i, centroids_len, boundaries);
            conditional_expectation(lo, hi, d)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Lloyd-Max core — Pure Integration (orchestrates operation helpers)
// ---------------------------------------------------------------------------

/// Perform one Lloyd-Max iteration step: compute boundaries, update centroids,
/// measure distortion, and check convergence.
///
/// Pure Integration: delegates to `midpoint_boundaries`, `update_centroids`,
/// `compute_distortion`, and `has_converged`.  Returns the new centroids,
/// the new distortion, and a convergence flag.
fn lloyd_max_step(centroids: &[f64], prev_distortion: f64, d: usize) -> (Vec<f64>, f64, bool) {
    let boundaries = midpoint_boundaries(centroids);
    let new_centroids = update_centroids(centroids.len(), &boundaries, d);
    let distortion = compute_distortion(&new_centroids, &boundaries, d);
    let converged = has_converged(prev_distortion, distortion);
    (new_centroids, distortion, converged)
}

/// Run Lloyd-Max iterations starting from the given initial `centroids` for
/// dimension `d`.  Returns the converged [`Codebook`].
///
/// Pure Integration: delegates each iteration to `lloyd_max_step` and
/// final boundary computation to `midpoint_boundaries`.
fn lloyd_max_iterate(mut centroids: Vec<f64>, d: usize) -> Codebook {
    let mut prev_distortion = f64::MAX;

    converge(MAX_ITERATIONS, || {
        let (new_centroids, distortion, converged) = lloyd_max_step(&centroids, prev_distortion, d);
        centroids = new_centroids;
        prev_distortion = distortion;
        converged
    });

    let boundaries = midpoint_boundaries(&centroids);
    Codebook {
        centroids,
        boundaries,
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run the Lloyd-Max algorithm from scratch for arbitrary `(bits, dim)`.
///
/// Pure Integration: delegates centroid count to `centroid_count`,
/// initialization to `initialize_centroids`, and iteration to
/// `lloyd_max_iterate`.
pub fn generate_codebook(bits: u8, dim: usize) -> Codebook {
    let k = centroid_count(bits);
    let centroids = initialize_centroids(k);
    lloyd_max_iterate(centroids, dim)
}

// ---------------------------------------------------------------------------
// Unit tests for generation helpers
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // -- Named constants for test parameters --------------------------------

    /// Dimension used in integration / beta-PDF tests.
    const TEST_DIM: usize = 128;
    /// Number of centroids when using 3-bit quantization (2^3).
    const TEST_CENTROIDS_8: usize = 8;
    /// Number of centroids when using 4-bit quantization (2^4).
    const TEST_CENTROIDS_16: usize = 16;
    /// Bit width for 3-bit quantization in tests.
    const TEST_BITS_3: u8 = 3;
    /// Dimension 64 used in generate_codebook tests.
    const TEST_DIM_64: usize = 64;
    /// Numerator used in select_conditional_or_midpoint tests.
    const TEST_NUMERATOR: f64 = 3.0;
    /// Normal-case denominator for select_conditional_or_midpoint test.
    const TEST_DENOMINATOR: f64 = 2.0;
    /// Near-zero denominator for select_conditional_or_midpoint fallback test.
    const TEST_NEAR_ZERO_DENOM: f64 = 1e-31;

    // -- initialize_centroids -----------------------------------------------

    #[test]
    fn initialize_centroids_correct_count() {
        assert_eq!(
            initialize_centroids(TEST_CENTROIDS_8).len(),
            TEST_CENTROIDS_8
        );
        assert_eq!(
            initialize_centroids(TEST_CENTROIDS_16).len(),
            TEST_CENTROIDS_16
        );
    }

    #[test]
    fn initialize_centroids_sorted() {
        let c = initialize_centroids(TEST_CENTROIDS_8);
        for w in c.windows(2) {
            assert!(w[0] < w[1]);
        }
    }

    #[test]
    fn initialize_centroids_symmetric() {
        let c = initialize_centroids(TEST_CENTROIDS_8);
        let half = TEST_CENTROIDS_8 / 2;
        for i in 0..half {
            assert_relative_eq!(c[i], -c[TEST_CENTROIDS_8 - 1 - i], epsilon = 1e-14);
        }
    }

    #[test]
    fn initialize_centroids_within_support() {
        let c = initialize_centroids(TEST_CENTROIDS_16);
        for &v in &c {
            assert!(v > SUPPORT_MIN && v < SUPPORT_MAX);
        }
    }

    // -- midpoint_boundaries ------------------------------------------------

    #[test]
    fn midpoint_boundaries_correct_values() {
        let centroids = vec![-0.5, 0.0, 0.5];
        let b = midpoint_boundaries(&centroids);
        assert_eq!(b.len(), 2);
        assert_relative_eq!(b[0], -0.25, epsilon = 1e-14);
        assert_relative_eq!(b[1], 0.25, epsilon = 1e-14);
    }

    // -- bin_lower_bound / bin_upper_bound -----------------------------------

    #[test]
    fn bin_lower_bound_first() {
        let boundaries = vec![0.0];
        assert_relative_eq!(
            bin_lower_bound(0, &boundaries),
            SUPPORT_MIN,
            epsilon = 1e-15
        );
    }

    #[test]
    fn bin_lower_bound_second() {
        let boundaries = vec![0.0];
        assert_relative_eq!(bin_lower_bound(1, &boundaries), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn bin_upper_bound_last() {
        let boundaries = vec![0.0];
        assert_relative_eq!(
            bin_upper_bound(1, 2, &boundaries),
            SUPPORT_MAX,
            epsilon = 1e-15
        );
    }

    #[test]
    fn bin_upper_bound_first() {
        let boundaries = vec![0.0];
        assert_relative_eq!(bin_upper_bound(0, 2, &boundaries), 0.0, epsilon = 1e-15);
    }

    // -- has_converged ------------------------------------------------------

    #[test]
    fn has_converged_identical_values() {
        assert!(has_converged(1.0, 1.0));
    }

    #[test]
    fn has_converged_large_change() {
        assert!(!has_converged(1.0, 0.5));
    }

    // -- select_conditional_or_midpoint -------------------------------------

    #[test]
    fn select_conditional_or_midpoint_normal_case() {
        let result = select_conditional_or_midpoint(TEST_NUMERATOR, TEST_DENOMINATOR, 0.0, 1.0);
        assert_relative_eq!(result, TEST_NUMERATOR / TEST_DENOMINATOR, epsilon = 1e-15);
    }

    #[test]
    fn select_conditional_or_midpoint_near_zero_denom() {
        let result = select_conditional_or_midpoint(TEST_NUMERATOR, TEST_NEAR_ZERO_DENOM, 0.0, 1.0);
        assert_relative_eq!(result, 0.5, epsilon = 1e-15);
    }

    // -- conditional_expectation --------------------------------------------

    #[test]
    fn conditional_expectation_symmetric_interval() {
        // E[X | X in [-1, 1]] should be 0 by symmetry.
        let result = conditional_expectation(SUPPORT_MIN, SUPPORT_MAX, TEST_DIM);
        assert_relative_eq!(result, 0.0, epsilon = 1e-8);
    }

    // -- compute_distortion -------------------------------------------------

    #[test]
    fn compute_distortion_nonnegative() {
        let centroids = vec![-0.5, 0.0, 0.5];
        let boundaries = vec![-0.25, 0.25];
        let d = compute_distortion(&centroids, &boundaries, TEST_DIM);
        assert!(d >= 0.0);
    }

    // -- update_centroids ---------------------------------------------------

    #[test]
    fn update_centroids_correct_count() {
        let boundaries = midpoint_boundaries(&initialize_centroids(TEST_CENTROIDS_8));
        let updated = update_centroids(TEST_CENTROIDS_8, &boundaries, TEST_DIM);
        assert_eq!(updated.len(), TEST_CENTROIDS_8);
    }

    #[test]
    fn update_centroids_within_support() {
        let boundaries = midpoint_boundaries(&initialize_centroids(TEST_CENTROIDS_8));
        let updated = update_centroids(TEST_CENTROIDS_8, &boundaries, TEST_DIM);
        for &c in &updated {
            assert!((SUPPORT_MIN..=SUPPORT_MAX).contains(&c));
        }
    }

    // -- generate_codebook --------------------------------------------------

    #[test]
    fn generate_codebook_valid_structure() {
        let cb = generate_codebook(TEST_BITS_3, TEST_DIM_64);
        assert_eq!(cb.centroids.len(), TEST_CENTROIDS_8);
        assert_eq!(cb.boundaries.len(), TEST_CENTROIDS_8 - 1);
        for w in cb.centroids.windows(2) {
            assert!(w[0] < w[1]);
        }
    }

    // -- lloyd_max_step -----------------------------------------------------

    #[test]
    fn lloyd_max_step_reduces_distortion() {
        let centroids = initialize_centroids(TEST_CENTROIDS_8);
        let boundaries = midpoint_boundaries(&centroids);
        let initial_dist = compute_distortion(&centroids, &boundaries, TEST_DIM);
        let (new_centroids, new_dist, _) = lloyd_max_step(&centroids, f64::MAX, TEST_DIM);
        // The new distortion should be <= initial (Lloyd-Max is monotonically improving).
        assert!(new_dist <= initial_dist + 1e-15);
        assert_eq!(new_centroids.len(), TEST_CENTROIDS_8);
    }
}
