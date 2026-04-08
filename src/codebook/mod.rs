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

use crate::error::{require, Result, TurboQuantError};
use crate::math::{ln_gamma, HALF};
use crate::packed::is_valid_bits;

// Re-export `generate_codebook` so that existing callers (including
// integration tests) can continue to import it from `codebook`.
pub use gen::generate_codebook;

/// Lower bound of the support interval [-1, 1].
pub(crate) const SUPPORT_MIN: f64 = -1.0;

/// Upper bound of the support interval [-1, 1].
pub(crate) const SUPPORT_MAX: f64 = 1.0;

/// Minimum dimension for which the Beta-type PDF is well-defined.
/// For d < 3 the exponent (d-3)/2 is negative and the distribution degenerates.
const MIN_DIMENSION_FOR_PDF: usize = 3;

/// The exponent offset in the Beta-type kernel: (d - 3) / 2.
const KERNEL_EXPONENT_OFFSET: f64 = 3.0;

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

/// Static description of a pre-computed codebook stored as `const` arrays.
struct StaticCodebook {
    centroids: &'static [f64],
    boundaries: &'static [f64],
}

impl StaticCodebook {
    /// Convert to an owned [`Codebook`].
    fn to_codebook(&self) -> Codebook {
        Codebook {
            centroids: self.centroids.to_vec(),
            boundaries: self.boundaries.to_vec(),
        }
    }
}

// ---------------------------------------------------------------------------
// Pre-computed codebook statics
// ---------------------------------------------------------------------------

// --- 2-bit codebooks (4 centroids, 3 boundaries) ---

static CODEBOOK_2BIT_D32: StaticCodebook = StaticCodebook {
    centroids: &CENTROIDS_2B_D32,
    boundaries: &BOUNDARIES_2B_D32,
};
static CODEBOOK_2BIT_D64: StaticCodebook = StaticCodebook {
    centroids: &CENTROIDS_2B_D64,
    boundaries: &BOUNDARIES_2B_D64,
};
static CODEBOOK_2BIT_D128: StaticCodebook = StaticCodebook {
    centroids: &CENTROIDS_2B_D128,
    boundaries: &BOUNDARIES_2B_D128,
};
static CODEBOOK_2BIT_D256: StaticCodebook = StaticCodebook {
    centroids: &CENTROIDS_2B_D256,
    boundaries: &BOUNDARIES_2B_D256,
};

// --- 3-bit codebooks (8 centroids, 7 boundaries) ---

static CODEBOOK_3BIT_D32: StaticCodebook = StaticCodebook {
    centroids: &CENTROIDS_3B_D32,
    boundaries: &BOUNDARIES_3B_D32,
};
static CODEBOOK_3BIT_D64: StaticCodebook = StaticCodebook {
    centroids: &CENTROIDS_3B_D64,
    boundaries: &BOUNDARIES_3B_D64,
};
static CODEBOOK_3BIT_D128: StaticCodebook = StaticCodebook {
    centroids: &CENTROIDS_3B_D128,
    boundaries: &BOUNDARIES_3B_D128,
};
static CODEBOOK_3BIT_D256: StaticCodebook = StaticCodebook {
    centroids: &CENTROIDS_3B_D256,
    boundaries: &BOUNDARIES_3B_D256,
};

// --- 4-bit codebooks (16 centroids, 15 boundaries) ---

static CODEBOOK_4BIT_D32: StaticCodebook = StaticCodebook {
    centroids: &CENTROIDS_4B_D32,
    boundaries: &BOUNDARIES_4B_D32,
};
static CODEBOOK_4BIT_D64: StaticCodebook = StaticCodebook {
    centroids: &CENTROIDS_4B_D64,
    boundaries: &BOUNDARIES_4B_D64,
};
static CODEBOOK_4BIT_D128: StaticCodebook = StaticCodebook {
    centroids: &CENTROIDS_4B_D128,
    boundaries: &BOUNDARIES_4B_D128,
};
static CODEBOOK_4BIT_D256: StaticCodebook = StaticCodebook {
    centroids: &CENTROIDS_4B_D256,
    boundaries: &BOUNDARIES_4B_D256,
};

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

/// Look up a pre-computed static codebook reference for the given
/// `(bits, dim)` pair, returning `None` if not found.
///
/// Pure Operation: only match logic, returns a reference.
fn lookup_static_codebook_ref(bits: u8, dim: usize) -> Option<&'static StaticCodebook> {
    match (bits, dim) {
        (2, 32) => Some(&CODEBOOK_2BIT_D32),
        (2, 64) => Some(&CODEBOOK_2BIT_D64),
        (2, 128) => Some(&CODEBOOK_2BIT_D128),
        (2, 256) => Some(&CODEBOOK_2BIT_D256),
        (3, 32) => Some(&CODEBOOK_3BIT_D32),
        (3, 64) => Some(&CODEBOOK_3BIT_D64),
        (3, 128) => Some(&CODEBOOK_3BIT_D128),
        (3, 256) => Some(&CODEBOOK_3BIT_D256),
        (4, 32) => Some(&CODEBOOK_4BIT_D32),
        (4, 64) => Some(&CODEBOOK_4BIT_D64),
        (4, 128) => Some(&CODEBOOK_4BIT_D128),
        (4, 256) => Some(&CODEBOOK_4BIT_D256),
        _ => None,
    }
}

/// Look up a pre-computed static codebook, returning `None` if the `(bits, dim)`
/// pair is not in the table.
///
/// Pure Integration: delegates lookup to `lookup_static_codebook_ref` and
/// conversion to `StaticCodebook::to_codebook`.
fn lookup_static_codebook(bits: u8, dim: usize) -> Option<Codebook> {
    let sc = lookup_static_codebook_ref(bits, dim)?;
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
// Pre-computed tables
// ---------------------------------------------------------------------------

// 2-bit, d=32
const CENTROIDS_2B_D32: [f64; 4] = [
    -0.2633194690486469,
    -0.07980195923950476,
    0.07980195923950384,
    0.2633194690486451,
];
const BOUNDARIES_2B_D32: [f64; 3] = [-0.1715607141440758, 0.0, 0.17156071414407448];

// 3-bit, d=32
const CENTROIDS_3B_D32: [f64; 8] = [
    -0.3662684197149945,
    -0.2324607608635176,
    -0.13175624361004853,
    -0.042851570066700616,
    0.04285157006670112,
    0.1317562436100487,
    0.23246076086351877,
    0.36626841971499513,
];
const BOUNDARIES_3B_D32: [f64; 7] = [
    -0.29936459028925605,
    -0.18210850223678307,
    -0.08730390683837458,
    0.0,
    0.08730390683837491,
    0.18210850223678374,
    0.29936459028925694,
];

// 4-bit, d=32
const CENTROIDS_4B_D32: [f64; 16] = [
    -0.45356484406403885,
    -0.3500919292405678,
    -0.276742371576412,
    -0.2163621312970215,
    -0.16307095286109716,
    -0.11401417528619187,
    -0.06749924527967034,
    -0.02235501645565892,
    0.022355016455658717,
    0.06749924527967022,
    0.11401417528619151,
    0.16307095286109707,
    0.21636213129702167,
    0.27674237157641146,
    0.35009192924056776,
    0.4535648440640396,
];
const BOUNDARIES_4B_D32: [f64; 15] = [
    -0.40182838665230336,
    -0.31341715040848994,
    -0.24655225143671677,
    -0.18971654207905933,
    -0.13854256407364451,
    -0.09075671028293111,
    -0.04492713086766463,
    0.0,
    0.044927130867664464,
    0.09075671028293086,
    0.1385425640736443,
    0.18971654207905936,
    0.24655225143671655,
    0.3134171504084896,
    0.4018283866523037,
];

// 2-bit, d=64
const CENTROIDS_2B_D64: [f64; 4] = [
    -0.18749689292196112,
    -0.05651489047635318,
    0.05651489047635313,
    0.18749689292196103,
];
const BOUNDARIES_2B_D64: [f64; 3] = [-0.12200589169915715, 0.0, 0.12200589169915708];

// 2-bit, d=128
const CENTROIDS_2B_D128: [f64; 4] = [
    -0.13304154846077318,
    -0.03999160906335877,
    0.03999160906335891,
    0.13304154846077348,
];
const BOUNDARIES_2B_D128: [f64; 3] = [-0.08651657876206598, 0.0, 0.086_516_578_762_066_2];

// 2-bit, d=256
const CENTROIDS_2B_D256: [f64; 4] = [
    -0.09423779913129633,
    -0.02828860721372146,
    0.02828860721372166,
    0.09423779913129664,
];
const BOUNDARIES_2B_D256: [f64; 3] = [-0.061_263_203_172_508_9, 0.0, 0.06126320317250915];

// 3-bit, d=64
const CENTROIDS_3B_D64: [f64; 8] = [
    -0.26391407457137683,
    -0.16616801009118487,
    -0.093_832_375_844_160_5,
    -0.03046922045737837,
    0.03046922045737837,
    0.093_832_375_844_160_5,
    0.16616801009118487,
    0.26391407457137683,
];
const BOUNDARIES_3B_D64: [f64; 7] = [
    -0.21504104233128085,
    -0.13000019296767268,
    -0.06215079815076943,
    0.0,
    0.06215079815076943,
    0.13000019296767268,
    0.21504104233128085,
];

// 3-bit, d=128
const CENTROIDS_3B_D128: [f64; 8] = [
    -0.18839728518004373,
    -0.11813986946554235,
    -0.06658568378325364,
    -0.02160433847349997,
    0.02160433847349997,
    0.06658568378325364,
    0.11813986946554235,
    0.18839728518004373,
];
const BOUNDARIES_3B_D128: [f64; 7] = [
    -0.15326857732279303,
    -0.09236277662439799,
    -0.044_095_011_128_376_8,
    0.0,
    0.044_095_011_128_376_8,
    0.09236277662439799,
    0.15326857732279303,
];

// 3-bit, d=256
const CENTROIDS_3B_D256: [f64; 8] = [
    -0.13385436276083063,
    -0.083_765_531_459_768_9,
    -0.04716676527922715,
    -0.01529750782483941,
    0.01529750782483941,
    0.04716676527922715,
    0.083_765_531_459_768_9,
    0.13385436276083063,
];
const BOUNDARIES_3B_D256: [f64; 7] = [
    -0.10880994711029976,
    -0.06546614836949802,
    -0.03123213655203328,
    0.0,
    0.03123213655203328,
    0.06546614836949802,
    0.10880994711029976,
];

// 4-bit, d=64
const CENTROIDS_4B_D64: [f64; 16] = [
    -0.33092994168409773,
    -0.25307088610074774,
    -0.19901983361887085,
    -0.15508179062990365,
    -0.11662310388676207,
    -0.08141753279040376,
    -0.04815672368589858,
    -0.015_941_930_352_081_4,
    0.015_941_930_352_081_4,
    0.04815672368589858,
    0.08141753279040376,
    0.11662310388676207,
    0.15508179062990365,
    0.19901983361887085,
    0.25307088610074774,
    0.33092994168409773,
];
const BOUNDARIES_4B_D64: [f64; 15] = [
    -0.292_000_413_892_422_7,
    -0.22604535985980928,
    -0.17705081212438725,
    -0.13585244725833287,
    -0.09902031833858291,
    -0.06478712823815116,
    -0.03204932701898999,
    0.0,
    0.03204932701898999,
    0.06478712823815116,
    0.09902031833858291,
    0.13585244725833287,
    0.17705081212438725,
    0.22604535985980928,
    0.292_000_413_892_422_7,
];

// 4-bit, d=128
const CENTROIDS_4B_D128: [f64; 16] = [
    -0.23777655506958537,
    -0.18096588552769086,
    -0.14193912272806147,
    -0.11041538921898804,
    -0.08293881469006784,
    -0.05785765497830671,
    -0.03420549908335103,
    -0.01132093590150223,
    0.01132093590150223,
    0.03420549908335103,
    0.05785765497830671,
    0.08293881469006784,
    0.11041538921898804,
    0.14193912272806147,
    0.18096588552769086,
    0.23777655506958537,
];
const BOUNDARIES_4B_D128: [f64; 15] = [
    -0.209_371_220_298_638_1,
    -0.16145250412787615,
    -0.12617725597352475,
    -0.09667710195452794,
    -0.07039823483418728,
    -0.04603157703082887,
    -0.02276321749242663,
    0.0,
    0.02276321749242663,
    0.04603157703082887,
    0.07039823483418728,
    0.09667710195452794,
    0.12617725597352475,
    0.16145250412787615,
    0.209_371_220_298_638_1,
];

// 4-bit, d=256
const CENTROIDS_4B_D256: [f64; 16] = [
    -0.16949853314441155,
    -0.12868871755030106,
    -0.10080108457584613,
    -0.07834675699488723,
    -0.05881658417438018,
    -0.04101444098641885,
    -0.02424206232116148,
    -0.00802245010411462,
    0.00802245010411462,
    0.02424206232116148,
    0.04101444098641885,
    0.05881658417438018,
    0.07834675699488723,
    0.10080108457584613,
    0.12868871755030106,
    0.16949853314441155,
];
const BOUNDARIES_4B_D256: [f64; 15] = [
    -0.14909362534735632,
    -0.114_744_901_063_073_6,
    -0.08957392078536669,
    -0.06858167058463371,
    -0.04991551258039952,
    -0.03262825165379016,
    -0.01613225621263805,
    0.0,
    0.01613225621263805,
    0.03262825165379016,
    0.04991551258039952,
    0.06858167058463371,
    0.08957392078536669,
    0.114_744_901_063_073_6,
    0.14909362534735632,
];

// ---------------------------------------------------------------------------
// Unit tests for codebook lookup and Beta PDF
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
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

    use crate::math::{ln_gamma, simpsons_integrate};

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
