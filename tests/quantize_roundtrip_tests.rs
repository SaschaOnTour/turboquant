//! Quantize roundtrip tests extracted from the former `roundtrip_tests.rs`.

use approx::assert_abs_diff_eq;
use turboquant::packed::TurboQuantConfig;
use turboquant::quantize::{dequantize_rotated, dequantize_vec, l2_norm, quantize_vec};
use turboquant::test_utils::pseudo_random_vec;

/// Default seed for rotation.
const TEST_SEED: u64 = 42;
/// Tolerance for norm comparisons after quantization roundtrip.
/// 3-bit quantization introduces ~18% relative error on average (sqrt(MSE=0.034)),
/// so the norm can deviate significantly.  f16 rounding adds further noise.
const NORM_EPSILON: f32 = 0.35;
/// Tolerance for near-zero checks.
const ZERO_EPSILON: f32 = 0.1;

/// Bit-widths covered by the parametric roundtrip tests.
const BITS: &[u8] = &[2, 3, 4];
/// Dimensions covered by the parametric roundtrip tests.
const DIMS: &[usize] = &[64, 128, 256];

fn squared_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum()
}

fn roundtrip_check(bits: u8, dim: usize, seed: u64) {
    let data = pseudo_random_vec(dim, seed);
    let config = TurboQuantConfig::new(bits, dim)
        .unwrap()
        .with_seed(TEST_SEED);
    let block = quantize_vec(&config, &data).unwrap();
    let recovered = dequantize_vec(&config, &block).unwrap();

    let orig_norm_sq = data.iter().map(|&x| x * x).sum::<f32>();
    let err_sq = squared_error(&data, &recovered);
    let relative_mse = err_sq / orig_norm_sq;

    // Single-vector relative MSE can be much higher than the aggregate mean
    // (0.034 for TQ3, 0.009 for TQ4, ~0.10 for TQ2). The proper quality gate
    // is `mse_validation` which checks over 10,000 vectors.
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

/// Name of the special-vector shape for diagnostic messages in
/// `special_vector_check`. Each variant carries its own absolute-norm bounds.
#[derive(Clone, Copy)]
enum SpecialVector {
    Null,
    Unit,
    Constant,
}

impl SpecialVector {
    fn data(self, dim: usize) -> Vec<f32> {
        match self {
            Self::Null => vec![0.0; dim],
            Self::Unit => {
                let mut v = vec![0.0; dim];
                v[0] = 1.0;
                v
            }
            Self::Constant => vec![0.5; dim],
        }
    }

    fn bounds(self, dim: usize) -> (f32, f32) {
        /// Lower bound for unit-vector recovered norm.
        const UNIT_MIN: f32 = 0.3;
        /// Upper bound for unit-vector recovered norm.
        const UNIT_MAX: f32 = 2.0;
        /// Minimum retained-energy ratio for a constant input.
        const CONSTANT_MIN_RATIO: f32 = 0.1;
        /// Maximum retained-energy ratio for a constant input.
        const CONSTANT_MAX_RATIO: f32 = 3.0;
        /// Value used to build the constant vector (`vec![CONSTANT_VALUE; dim]`).
        const CONSTANT_VALUE: f32 = 0.5;
        match self {
            Self::Null => (-1.0, ZERO_EPSILON),
            Self::Unit => (UNIT_MIN, UNIT_MAX),
            Self::Constant => {
                let orig = (dim as f32).sqrt() * CONSTANT_VALUE;
                (CONSTANT_MIN_RATIO * orig, CONSTANT_MAX_RATIO * orig)
            }
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::Unit => "unit",
            Self::Constant => "constant",
        }
    }
}

fn special_vector_check(bits: u8, dim: usize, shape: SpecialVector) {
    let data = shape.data(dim);
    let config = TurboQuantConfig::new(bits, dim)
        .unwrap()
        .with_seed(TEST_SEED);
    let block = quantize_vec(&config, &data).unwrap();
    let recovered = dequantize_vec(&config, &block).unwrap();
    let rec_norm = l2_norm(&recovered);
    let (min_norm, max_norm) = shape.bounds(dim);
    let label = shape.label();
    assert!(
        rec_norm > min_norm,
        "bits={bits}: {label} vector recovered norm {rec_norm} below {min_norm}"
    );
    assert!(
        rec_norm < max_norm,
        "bits={bits}: {label} vector recovered norm {rec_norm} above {max_norm}"
    );
}

fn determinism_check(bits: u8, dim: usize, seed: u64) {
    let data = pseudo_random_vec(dim, seed);
    let config = TurboQuantConfig::new(bits, dim)
        .unwrap()
        .with_seed(TEST_SEED);
    let block_a = quantize_vec(&config, &data).unwrap();
    let block_b = quantize_vec(&config, &data).unwrap();
    let rec_a = dequantize_vec(&config, &block_a).unwrap();
    let rec_b = dequantize_vec(&config, &block_b).unwrap();
    assert_eq!(
        rec_a, rec_b,
        "bits={bits}: quantization should be deterministic"
    );
}

fn dequantize_rotated_check(bits: u8, dim: usize, seed: u64) {
    let data = pseudo_random_vec(dim, seed);
    let config = TurboQuantConfig::new(bits, dim)
        .unwrap()
        .with_seed(TEST_SEED);
    let block = quantize_vec(&config, &data).unwrap();
    let full = dequantize_vec(&config, &block).unwrap();
    let rotated = dequantize_rotated(&config, &block).unwrap();

    assert_ne!(
        full, rotated,
        "bits={bits}: rotated and full dequantize should differ"
    );

    let full_norm = l2_norm(&full);
    let rotated_norm = l2_norm(&rotated);
    assert_abs_diff_eq!(full_norm, rotated_norm, epsilon = NORM_EPSILON);
}

// -----------------------------------------------------------------------
// Parametric tests across (bits, dim)
// -----------------------------------------------------------------------

#[test]
fn roundtrip_all_bits_and_dims() {
    // Distinct deterministic seed per (bits, dim) — no collisions across the grid.
    for (bi, &bits) in BITS.iter().enumerate() {
        for (di, &dim) in DIMS.iter().enumerate() {
            let seed = 1000 * (bi as u64 + 1) + 100 * (di as u64 + 1);
            roundtrip_check(bits, dim, seed);
        }
    }
}

#[test]
fn special_vectors_across_bit_widths() {
    for &bits in BITS {
        for shape in [
            SpecialVector::Null,
            SpecialVector::Unit,
            SpecialVector::Constant,
        ] {
            special_vector_check(bits, 128, shape);
        }
    }
}

#[test]
fn determinism_all_bits() {
    for (i, &bits) in BITS.iter().enumerate() {
        determinism_check(bits, 128, 11111 * (i as u64 + 1));
    }
}

#[test]
fn different_dimensions_all_bits() {
    for &bits in BITS {
        for &dim in DIMS {
            let config = TurboQuantConfig::new(bits, dim)
                .unwrap()
                .with_seed(TEST_SEED);
            let data = pseudo_random_vec(dim, dim as u64 + bits as u64 * 1000);
            let block = quantize_vec(&config, &data).unwrap();
            let recovered = dequantize_vec(&config, &block).unwrap();
            assert_eq!(recovered.len(), dim);
        }
    }
}

#[test]
fn dequantize_rotated_differs_but_same_norm_all_bits() {
    for (i, &bits) in BITS.iter().enumerate() {
        dequantize_rotated_check(bits, 128, 33333 * (i as u64 + 1));
    }
}

#[test]
fn packed_block_records_correct_bits_all_widths() {
    let seeds = [44444_u64, 55555, 66666];
    for (&bits, &seed) in BITS.iter().zip(seeds.iter()) {
        let config = TurboQuantConfig::new(bits, 64)
            .unwrap()
            .with_seed(TEST_SEED);
        let data = pseudo_random_vec(64, seed);
        let block = quantize_vec(&config, &data).unwrap();
        assert_eq!(block.bits, bits);
        let recovered = dequantize_vec(&config, &block).unwrap();
        assert_eq!(recovered.len(), 64);
    }
}

/// Cross-property smoke test: exercises every roundtrip quality helper
/// (MSE, special-vector norm bounds, rotated-vs-full) in one go. Binds
/// the check helpers into a single SRP cluster so the module reads as
/// one coherent "quantize roundtrip quality" responsibility.
#[test]
fn all_roundtrip_properties_smoke_test() {
    let bits = 3u8;
    let dim = 128usize;
    roundtrip_check(bits, dim, 42);
    special_vector_check(bits, dim, SpecialVector::Null);
    special_vector_check(bits, dim, SpecialVector::Unit);
    special_vector_check(bits, dim, SpecialVector::Constant);
    determinism_check(bits, dim, 1337);
    dequantize_rotated_check(bits, dim, 77);
}
