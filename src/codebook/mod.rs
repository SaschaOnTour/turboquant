//! Codebook lookup, Beta PDF, and pre-computed tables for Lloyd-Max quantization.
//!
//! After applying a random rotation to a *d*-dimensional unit vector, each
//! coordinate follows a Beta-type distribution on [-1, 1]:
//!
//! ```text
//! f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)
//! ```
//!
//! This module stores pre-computed Lloyd-Max codebooks for common `(bits, dim)`
//! pairs and provides lookup / nearest-centroid queries.  The generation
//! algorithm itself lives in the [`gen`] sub-module.

mod gen;
mod tables;

use crate::error::{require, Result, TurboQuantError};
use crate::packed::is_valid_bits;

// Re-export `generate_codebook` and `beta_pdf` so that existing callers
// (including integration tests) can continue to import them from `codebook`.
pub use gen::{beta_pdf, generate_codebook};

/// Lower bound of the support interval [-1, 1].
pub(crate) const SUPPORT_MIN: f64 = -1.0;

/// Upper bound of the support interval [-1, 1].
pub(crate) const SUPPORT_MAX: f64 = 1.0;

// ---------------------------------------------------------------------------
// Codebook struct
// ---------------------------------------------------------------------------

/// A static codebook: sorted centroids and the decision boundaries between
/// them.  For *k* centroids there are *k-1* interior boundaries; the outer
/// boundaries are implicitly -1 and +1.
#[derive(Debug, Clone)]
pub struct Codebook {
    /// Sorted centroid values (length = 2^bits).
    pub centroids: Vec<f64>,
    /// Interior decision boundaries (length = 2^bits - 1).
    pub boundaries: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Pure Operation: Validation helpers (logic only, no calls)
// ---------------------------------------------------------------------------

/// Validate the bit width, returning an error if unsupported.
///
/// Pure Integration: only calls `require` and `is_valid_bits` (from `packed`).
fn validate_bits(bits: u8) -> Result<()> {
    require(is_valid_bits(bits), TurboQuantError::UnsupportedBits(bits))
}

/// Compute the number of centroids for a given bit width: 2^bits.
pub(crate) fn centroid_count(bits: u8) -> usize {
    1usize << bits
}

// ---------------------------------------------------------------------------
// Pure Operation: nearest centroid lookup
// ---------------------------------------------------------------------------

/// Binary search over interior boundaries to find the bin index for `value`.
fn boundary_binary_search(value: f64, boundaries: &[f64]) -> u8 {
    let mut lo: usize = 0;
    let mut hi: usize = boundaries.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if value > boundaries[mid] {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo as u8
}

/// Find the index of the nearest centroid for a scalar `value`.
///
/// Uses the interior boundaries for an O(log k) binary search.
///
/// Pure Integration: delegates to `boundary_binary_search`.
pub fn nearest_centroid(value: f64, codebook: &Codebook) -> u8 {
    boundary_binary_search(value, &codebook.boundaries)
}

// ---------------------------------------------------------------------------
// Pure Operation: static codebook lookup
// ---------------------------------------------------------------------------

/// Look up a pre-computed static codebook, returning `None` if the `(bits, dim)`
/// pair is not in the table.
///
/// Pure Integration: delegates lookup to `tables::lookup_static_codebook_ref` and
/// conversion to `StaticCodebook::to_codebook`.
fn lookup_static_codebook(bits: u8, dim: usize) -> Option<Codebook> {
    let sc = tables::lookup_static_codebook_ref(bits, dim)?;
    Some(sc.to_codebook())
}

// ---------------------------------------------------------------------------
// Public API — Pure Integration functions (orchestrate other fns)
// ---------------------------------------------------------------------------

/// Return a pre-computed [`Codebook`] for the given `(bits, dim)` pair.
///
/// Falls back to computing one on the fly if the pair is not in the
/// pre-computed table.
///
/// # Errors
///
/// Returns [`TurboQuantError::UnsupportedBits`] if `bits` is not 2, 3, or 4.
///
/// Pure Integration: delegates validation to `validate_bits`, lookup to
/// `lookup_static_codebook`, and generation to `gen::generate_codebook`.
pub fn get_codebook(bits: u8, dim: usize) -> Result<Codebook> {
    validate_bits(bits)?;
    let maybe = lookup_static_codebook(bits, dim);
    Ok(maybe.unwrap_or_else(|| gen::generate_codebook(bits, dim)))
}

// ---------------------------------------------------------------------------
// Unit tests for codebook lookup and Beta PDF
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::tables::*;
    use super::*;
    use approx::assert_relative_eq;

    // -- Named constants for test parameters --------------------------------

    /// Dimension used in integration / beta-PDF tests.
    const TEST_DIM: usize = 128;
    /// Number of steps for Simpson's rule integration.
    const INTEGRATION_STEPS: usize = 2048;
    /// Number of centroids when using 3-bit quantization (2^3).
    const TEST_CENTROIDS_8: usize = 8;
    /// Number of centroids when using 4-bit quantization (2^4).
    const TEST_CENTROIDS_16: usize = 16;
    /// Bit width for 3-bit quantization in tests.
    const TEST_BITS_3: u8 = 3;
    /// Bit width for 4-bit quantization in tests.
    const TEST_BITS_4: u8 = 4;
    /// Dimensions used in beta_pdf symmetry and boundary tests.
    const TEST_DIMS: [usize; 3] = [64, 128, 256];
    /// X-values used in beta_pdf symmetry tests.
    const TEST_X_VALUES: [f64; 5] = [0.0, 0.1, 0.3, 0.5, 0.9];
    /// Number of centroids when using 2-bit quantization (2^2).
    const TEST_CENTROIDS_4: usize = 4;
    /// Bit width for 2-bit quantization in tests.
    const TEST_BITS_2: u8 = 2;
    /// Known `(bits, dim)` pairs used in lookup tests.
    const KNOWN_CODEBOOK_CONFIGS: [(u8, usize); 12] = [
        (2, 32),
        (2, 64),
        (2, 128),
        (2, 256),
        (3, 32),
        (3, 64),
        (3, 128),
        (3, 256),
        (4, 32),
        (4, 64),
        (4, 128),
        (4, 256),
    ];

    // -- Test-only helper functions (inlined out of production code) ---------

    use crate::math::{ln_gamma, simpsons_integrate, HALF};

    /// Compute the log-normalization constant for the Beta-type PDF (test-only).
    fn beta_pdf_log_normalization(df: f64) -> f64 {
        let half_df = df * HALF;
        let half_df_minus_one = (df - 1.0) * HALF;
        let half_ln_pi = HALF * core::f64::consts::PI.ln();
        ln_gamma(half_df) - half_ln_pi - ln_gamma(half_df_minus_one)
    }

    /// Compute the midpoint of an interval (test-only).
    fn interval_midpoint(a: f64, b: f64) -> f64 {
        (a + b) * HALF
    }

    /// Small epsilon to guard against division by near-zero values (test-only).
    const EPSILON_ZERO: f64 = 1e-30;

    /// Check whether a denominator is too small for safe division (test-only).
    fn is_near_zero(value: f64) -> bool {
        value.abs() < EPSILON_ZERO
    }

    // -- beta_pdf -----------------------------------------------------------

    #[test]
    fn beta_pdf_integrates_to_approximately_one() {
        let d = TEST_DIM;
        let integral = simpsons_integrate(
            |x| beta_pdf(x, d),
            SUPPORT_MIN,
            SUPPORT_MAX,
            INTEGRATION_STEPS,
        );
        assert_relative_eq!(integral, 1.0, epsilon = 1e-4);
    }

    #[test]
    fn beta_pdf_is_symmetric() {
        for d in TEST_DIMS {
            for &x in &TEST_X_VALUES {
                assert_relative_eq!(beta_pdf(x, d), beta_pdf(-x, d), epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn beta_pdf_zero_at_boundary() {
        for d in TEST_DIMS {
            assert_relative_eq!(beta_pdf(SUPPORT_MAX, d), 0.0, epsilon = 1e-15);
            assert_relative_eq!(beta_pdf(SUPPORT_MIN, d), 0.0, epsilon = 1e-15);
        }
    }

    #[test]
    fn beta_pdf_zero_outside_support() {
        assert_relative_eq!(beta_pdf(1.5, TEST_DIM), 0.0, epsilon = 1e-15);
        assert_relative_eq!(beta_pdf(-2.0, TEST_DIM), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn beta_pdf_zero_for_low_dimension() {
        assert_relative_eq!(beta_pdf(0.0, 2), 0.0, epsilon = 1e-15);
        assert_relative_eq!(beta_pdf(0.0, 1), 0.0, epsilon = 1e-15);
    }

    // -- boundary_binary_search ---------------------------------------------

    #[test]
    fn boundary_binary_search_first_bin() {
        let boundaries = vec![-0.5, 0.0, 0.5];
        assert_eq!(boundary_binary_search(-0.9, &boundaries), 0);
    }

    #[test]
    fn boundary_binary_search_last_bin() {
        let boundaries = vec![-0.5, 0.0, 0.5];
        assert_eq!(boundary_binary_search(0.9, &boundaries), 3);
    }

    #[test]
    fn boundary_binary_search_middle() {
        let boundaries = vec![-0.5, 0.0, 0.5];
        assert_eq!(boundary_binary_search(-0.1, &boundaries), 1);
        assert_eq!(boundary_binary_search(0.1, &boundaries), 2);
    }

    // -- interval_midpoint --------------------------------------------------

    #[test]
    fn interval_midpoint_basic() {
        assert_relative_eq!(interval_midpoint(0.0, 1.0), 0.5, epsilon = 1e-15);
        assert_relative_eq!(interval_midpoint(-1.0, 1.0), 0.0, epsilon = 1e-15);
    }

    // -- is_near_zero -------------------------------------------------------

    #[test]
    fn is_near_zero_true_for_tiny() {
        assert!(is_near_zero(1e-31));
        assert!(is_near_zero(-1e-31));
        assert!(is_near_zero(0.0));
    }

    #[test]
    fn is_near_zero_false_for_normal() {
        assert!(!is_near_zero(1e-10));
        assert!(!is_near_zero(-0.001));
    }

    // -- lookup_static_codebook / lookup_static_codebook_ref ----------------

    #[test]
    fn lookup_known_configs_return_some() {
        for &(bits, dim) in &KNOWN_CODEBOOK_CONFIGS {
            assert!(
                lookup_static_codebook_ref(bits, dim).is_some(),
                "expected Some for ({bits}, {dim})"
            );
            assert!(lookup_static_codebook(bits, dim).is_some());
        }
    }

    #[test]
    fn lookup_unknown_config_returns_none() {
        assert!(lookup_static_codebook_ref(3, 512).is_none());
        assert!(lookup_static_codebook(4, 16).is_none());
    }

    // -- centroid_count -----------------------------------------------------

    #[test]
    fn centroid_count_3bit() {
        assert_eq!(centroid_count(TEST_BITS_3), TEST_CENTROIDS_8);
    }

    #[test]
    fn centroid_count_4bit() {
        assert_eq!(centroid_count(TEST_BITS_4), TEST_CENTROIDS_16);
    }

    // -- to_codebook --------------------------------------------------------

    #[test]
    fn to_codebook_copies_correctly() {
        let sc = &CODEBOOK_3BIT_D64;
        let cb = sc.to_codebook();
        assert_eq!(cb.centroids.len(), sc.centroids.len());
        assert_eq!(cb.boundaries.len(), sc.boundaries.len());
        for (a, b) in cb.centroids.iter().zip(sc.centroids.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-15);
        }
        for (a, b) in cb.boundaries.iter().zip(sc.boundaries.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-15);
        }
    }

    // -- validate_bits / is_valid_bits (from packed) -------------------------

    #[test]
    fn validate_bits_accepts_2_3_and_4() {
        assert!(validate_bits(TEST_BITS_2).is_ok());
        assert!(validate_bits(TEST_BITS_3).is_ok());
        assert!(validate_bits(TEST_BITS_4).is_ok());
    }

    #[test]
    fn validate_bits_rejects_others() {
        assert!(validate_bits(0).is_err());
        assert!(validate_bits(1).is_err());
        assert!(validate_bits(5).is_err());
    }

    // -- centroid_count 2-bit -----------------------------------------------

    #[test]
    fn centroid_count_2bit() {
        assert_eq!(centroid_count(TEST_BITS_2), TEST_CENTROIDS_4);
    }

    // -- beta_pdf_log_normalization -----------------------------------------

    #[test]
    fn beta_pdf_log_normalization_positive_for_high_d() {
        // For d=128, the normalization constant should be > 1 (concentrated PDF),
        // so the log should be positive.
        let ln_norm = beta_pdf_log_normalization(TEST_DIM as f64);
        assert!(ln_norm > 0.0);
    }
}
