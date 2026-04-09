//! Core quantize / dequantize pipeline.
//!
//! Wires together [`codebook`], [`rotation`], and [`packed`] to provide the
//! full TurboQuant quantization and dequantization path.
//!
//! ## Quantize path
//!
//! ```text
//! Input: &[f32]  (key or value vector, length d)
//! 1. scale = L2-norm(x)
//! 2. x_normalized = x / scale
//! 3. x_rotated = rotate(x_normalized)        // WHT + sign-flips
//! 4. indices[i] = nearest_centroid(x_rotated[i], codebook)
//! 5. packed = PackedBlock::new(bits, scale, indices)
//! Output: PackedBlock
//! ```
//!
//! ## Dequantize path
//!
//! ```text
//! Input: PackedBlock
//! 1. indices = block.unpack(dim)
//! 2. x_rotated[i] = codebook.centroids[indices[i]]
//! 3. x_normalized = inverse_rotate(x_rotated)
//! 4. x = x_normalized * scale
//! Output: Vec<f32>
//! ```

use half::f16;

use crate::codebook::{get_codebook, nearest_centroid, Codebook};
use crate::error::{check_values_match, Result};
use crate::packed::{PackedBlock, TurboQuantConfig};
use crate::rotation::{generate_sign_pattern, rotate, RotationOrder};

// ---------------------------------------------------------------------------
// Named constants (no magic numbers)
// ---------------------------------------------------------------------------

/// Minimum norm below which a vector is treated as zero to avoid division
/// by near-zero values.
const MIN_NORM: f32 = 1e-10;

// ---------------------------------------------------------------------------
// Pure Operation helpers (logic only, no calls to other project functions)
// ---------------------------------------------------------------------------

/// Computes the L2 norm of a vector.
///
/// Pure Operation: arithmetic only.
pub fn l2_norm(data: &[f32]) -> f32 {
    data.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

/// Divides every element of `data` by `norm`, in place.
///
/// If `norm` is below [`MIN_NORM`], all elements are set to zero.
///
/// Pure Operation: arithmetic only.
pub fn normalize_inplace(data: &mut [f32], norm: f32) {
    if norm < MIN_NORM {
        for v in data.iter_mut() {
            *v = 0.0;
        }
    } else {
        let inv = 1.0 / norm;
        for v in data.iter_mut() {
            *v *= inv;
        }
    }
}

/// Multiplies every element of `data` by `factor`, in place.
///
/// Pure Operation: arithmetic only.
pub fn scale_inplace(data: &mut [f32], factor: f32) {
    for v in data.iter_mut() {
        *v *= factor;
    }
}

/// Maps each f32 coordinate to its nearest centroid index using binary search
/// on the codebook boundaries.
///
/// Pure Operation: iterates over coordinates, delegates each lookup to
/// [`nearest_centroid`] which is a pure leaf function.
pub fn quantize_coordinates(rotated: &[f32], codebook: &Codebook) -> Vec<u8> {
    rotated
        .iter()
        .map(|&v| nearest_centroid(v as f64, codebook))
        .collect()
}

/// Maps centroid indices back to their f32 centroid values.
///
/// Pure Operation: index lookup only.
pub fn lookup_centroids(indices: &[u8], codebook: &Codebook) -> Vec<f32> {
    indices
        .iter()
        .map(|&idx| codebook.centroids[idx as usize] as f32)
        .collect()
}

/// Maps centroid indices into a caller-provided buffer, avoiding allocation.
///
/// Hot-path variant of [`lookup_centroids`]: reuses `out` across repeated
/// calls to eliminate per-key allocations in attention loops.
///
/// Pure Operation: index lookup only.
pub fn lookup_centroids_into(indices: &[u8], codebook: &Codebook, out: &mut Vec<f32>) {
    out.clear();
    out.extend(
        indices
            .iter()
            .map(|&idx| codebook.centroids[idx as usize] as f32),
    );
}

/// Selects the scale factor: zero if the norm is negligible, otherwise the norm
/// converted to f16.
///
/// Pure Operation: comparison and external conversion only, no project calls.
fn select_scale(norm: f32) -> f16 {
    let effective = if norm < MIN_NORM { 0.0 } else { norm };
    f16::from_f32(effective)
}

// ---------------------------------------------------------------------------
// Integration: quantize_vec
// ---------------------------------------------------------------------------

/// Quantizes a floating-point vector into a packed [`PackedBlock`].
///
/// Pure Integration: orchestrates `check_values_match`, `get_codebook`,
/// `generate_sign_pattern`, `l2_norm`, `normalize_inplace` (handles zero-norm
/// internally), `rotate`, `quantize_coordinates`, `select_scale`, and
/// `PackedBlock::new`.
///
/// # Errors
///
/// Returns [`TurboQuantError::DimensionMismatch`] if `data.len() != config.dim`.
pub fn quantize_vec(config: &TurboQuantConfig, data: &[f32]) -> Result<PackedBlock> {
    check_values_match(data.len(), config.dim)?;

    let codebook = get_codebook(config.bits, config.dim)?;
    let sign_pattern = generate_sign_pattern(config.dim, config.rotation_seed);
    let norm = l2_norm(data);

    let mut working = data.to_vec();
    normalize_inplace(&mut working, norm);
    rotate(&mut working, &sign_pattern, RotationOrder::Forward)?;

    let indices = quantize_coordinates(&working, &codebook);
    let scale = select_scale(norm);

    Ok(PackedBlock::new(config.bits, scale, &indices))
}

/// Quantizes a floating-point vector into a packed [`PackedBlock`] using
/// pre-fetched codebook and sign pattern.
///
/// Hot-path variant of [`quantize_vec`]: avoids repeated codebook allocation
/// and sign-pattern generation when quantizing many vectors with the same
/// config (e.g. in batch KV-cache insertion during prefill).
///
/// Integration: orchestrates `check_values_match`, `l2_norm`,
/// `normalize_inplace`, `rotate`, `quantize_coordinates`, `select_scale`,
/// and `PackedBlock::new` -- all using caller-provided codebook and sign
/// pattern.
///
/// # Errors
///
/// Returns [`TurboQuantError::DimensionMismatch`] if `data.len() != config.dim`.
pub fn quantize_vec_with_codebook(
    config: &TurboQuantConfig,
    data: &[f32],
    codebook: &Codebook,
    sign_pattern: &[f32],
) -> Result<PackedBlock> {
    check_values_match(data.len(), config.dim)?;

    let norm = l2_norm(data);

    let mut working = data.to_vec();
    normalize_inplace(&mut working, norm);
    rotate(&mut working, sign_pattern, RotationOrder::Forward)?;

    let indices = quantize_coordinates(&working, codebook);
    let scale = select_scale(norm);

    Ok(PackedBlock::new(config.bits, scale, &indices))
}

// ---------------------------------------------------------------------------
// Integration: dequantize_vec
// ---------------------------------------------------------------------------

/// Dequantizes a [`PackedBlock`] back into a floating-point vector.
///
/// Integration: unpacks indices, looks up centroids, applies inverse rotation,
/// and scales by the stored norm.
///
/// # Errors
///
/// Returns an error if the inverse rotation fails (should not happen if the
/// block was produced by [`quantize_vec`] with valid config).
pub fn dequantize_vec(config: &TurboQuantConfig, block: &PackedBlock) -> Result<Vec<f32>> {
    let codebook = get_codebook(config.bits, config.dim)?;
    let sign_pattern = generate_sign_pattern(config.dim, config.rotation_seed);
    dequantize_vec_with_codebook(config, block, &codebook, &sign_pattern)
}

/// Dequantizes a [`PackedBlock`] using a pre-fetched codebook and sign pattern.
///
/// Hot-path variant: avoids repeated codebook allocation and sign-pattern
/// generation when dequantizing many blocks with the same config (e.g. in
/// attention score computation).
///
/// Integration: unpacks indices, looks up centroids, applies inverse rotation,
/// and scales by the stored norm.
///
/// # Errors
///
/// Returns an error if the inverse rotation fails.
pub fn dequantize_vec_with_codebook(
    config: &TurboQuantConfig,
    block: &PackedBlock,
    codebook: &Codebook,
    sign_pattern: &[f32],
) -> Result<Vec<f32>> {
    let indices = block.unpack(config.dim);
    let mut reconstructed = lookup_centroids(&indices, codebook);

    rotate(&mut reconstructed, sign_pattern, RotationOrder::Inverse)?;

    let scale = block.scale.to_f32();
    scale_inplace(&mut reconstructed, scale);

    Ok(reconstructed)
}

/// Dequantizes a [`PackedBlock`] into a caller-provided buffer, avoiding
/// allocation on the hot path.
///
/// Uses pre-fetched codebook and sign pattern, plus caller-owned scratch
/// buffers for indices and output.  Designed for tight loops (attention
/// score / weighted value computation).
///
/// Integration: unpacks indices, looks up centroids, applies inverse rotation,
/// and scales by the stored norm -- all into caller-provided buffers.
///
/// # Errors
///
/// Returns an error if the inverse rotation fails.
pub fn dequantize_into_with_codebook(
    config: &TurboQuantConfig,
    block: &PackedBlock,
    codebook: &Codebook,
    sign_pattern: &[f32],
    scratch: &mut DequantScratch,
) -> Result<()> {
    block.unpack_into(config.dim, &mut scratch.indices);
    lookup_centroids_into(&scratch.indices, codebook, &mut scratch.values);
    rotate(&mut scratch.values, sign_pattern, RotationOrder::Inverse)?;
    scale_inplace(&mut scratch.values, block.scale.to_f32());
    Ok(())
}

/// Pre-allocated scratch buffers for hot-path dequantization.
///
/// Avoids per-key heap allocation in `dequantize_into_with_codebook`.
pub struct DequantScratch {
    /// Buffer for unpacked indices.
    pub(crate) indices: Vec<u8>,
    /// Buffer for reconstructed f32 values.
    pub(crate) values: Vec<f32>,
}

impl DequantScratch {
    /// Creates scratch buffers pre-allocated for the given dimension.
    pub fn new(dim: usize) -> Self {
        Self {
            indices: Vec::with_capacity(dim),
            values: Vec::with_capacity(dim),
        }
    }
}

// ---------------------------------------------------------------------------
// Integration: dequantize_rotated
// ---------------------------------------------------------------------------

/// Dequantizes a [`PackedBlock`] but *skips* the inverse rotation.
///
/// Returns the reconstructed vector in the **rotated domain**.  This is used
/// by the attention optimization (Phase A6) where queries are pre-rotated so
/// that the dot product can be computed directly in rotated space.
///
/// Integration: unpacks indices, looks up centroids, scales -- but does NOT
/// call `inverse_rotate`.
///
/// # Errors
///
/// Returns an error if the codebook lookup fails.
// qual:api -- public API for rotated-domain attention optimization
pub fn dequantize_rotated(config: &TurboQuantConfig, block: &PackedBlock) -> Result<Vec<f32>> {
    let codebook = get_codebook(config.bits, config.dim)?;

    let indices = block.unpack(config.dim);
    let mut reconstructed = lookup_centroids(&indices, &codebook);

    let scale = block.scale.to_f32();
    scale_inplace(&mut reconstructed, scale);

    Ok(reconstructed)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packed::{BITS_TQ2, BITS_TQ3, BITS_TQ4};
    use crate::test_utils::pseudo_random_vec;

    /// Test dimension for unit tests.
    const TEST_DIM: usize = 64;
    /// Small dimension for block-construction tests.
    const TEST_SMALL_DIM: usize = 8;
    /// Test seed for reproducible sign patterns.
    const TEST_SEED: u64 = 42;
    /// Seed for 3-bit roundtrip test.
    const TEST_SEED_3BIT: u64 = 12345;
    /// Seed for 4-bit roundtrip test.
    const TEST_SEED_4BIT: u64 = 54321;
    /// Seed for determinism test.
    const TEST_SEED_DETERM: u64 = 99999;
    /// Seed for rotated-domain test.
    const TEST_SEED_ROTATED: u64 = 77777;
    /// Tolerance for floating-point comparisons.
    const FLOAT_EPSILON: f32 = 1e-6;
    /// Tolerance for norm comparisons (f16 introduces rounding).
    const NORM_EPSILON: f32 = 0.02;
    /// Maximum acceptable relative error for a single-vector roundtrip.
    /// Individual vectors can have much higher error than the aggregate MSE;
    /// the proper quality gate is in mse_validation.rs.
    const MAX_SINGLE_VEC_RELATIVE_ERROR: f32 = 1.0;
    /// Scale value for known-norm test (||[3,4]|| = 5).
    const TEST_SCALE_VALUE: f32 = 5.0;
    /// Norm value used in normalize and scale tests.
    const TEST_NORM_VALUE: f32 = 2.0;
    /// Constant value A for block scale tests.
    const TEST_CONST_VAL_A: f32 = 2.5;
    /// Constant value B for block scale tests.
    const TEST_CONST_VAL_B: f32 = 3.0;

    // -- l2_norm --------------------------------------------------------------

    #[test]
    fn l2_norm_of_unit_vector() {
        let mut v = vec![0.0_f32; TEST_DIM];
        v[0] = 1.0;
        let norm = l2_norm(&v);
        assert!((norm - 1.0).abs() < FLOAT_EPSILON);
    }

    #[test]
    fn l2_norm_of_zero_vector() {
        let v = vec![0.0_f32; TEST_DIM];
        let norm = l2_norm(&v);
        assert!(norm < FLOAT_EPSILON);
    }

    #[test]
    fn l2_norm_of_known_vector() {
        // [3, 4] -> norm = 5
        let v = vec![3.0_f32, 4.0];
        let norm = l2_norm(&v);
        assert!((norm - TEST_SCALE_VALUE).abs() < FLOAT_EPSILON);
    }

    // -- normalize_inplace ----------------------------------------------------

    #[test]
    fn normalize_inplace_unit_result() {
        let mut v = vec![3.0_f32, 4.0];
        normalize_inplace(&mut v, TEST_SCALE_VALUE);
        assert!((v[0] - 0.6).abs() < FLOAT_EPSILON);
        assert!((v[1] - 0.8).abs() < FLOAT_EPSILON);
    }

    #[test]
    fn normalize_inplace_zero_norm_gives_zeros() {
        let mut v = vec![1.0_f32, 2.0, 3.0];
        normalize_inplace(&mut v, 0.0);
        for &val in &v {
            assert!(val.abs() < FLOAT_EPSILON);
        }
    }

    // -- scale_inplace --------------------------------------------------------

    #[test]
    fn scale_inplace_doubles() {
        let mut v = vec![1.0_f32, 2.0, 3.0];
        scale_inplace(&mut v, TEST_NORM_VALUE);
        assert!((v[0] - 2.0).abs() < FLOAT_EPSILON);
        assert!((v[1] - 4.0).abs() < FLOAT_EPSILON);
        assert!((v[2] - 6.0).abs() < FLOAT_EPSILON);
    }

    // -- values_match (from error.rs) / is_zero_norm --------------------------

    #[test]
    fn values_match_true() {
        assert!(crate::error::values_match(128, 128));
    }

    #[test]
    fn values_match_false() {
        assert!(!crate::error::values_match(64, 128));
    }

    #[test]
    fn select_scale_zero_for_tiny_norm() {
        assert_eq!(select_scale(1e-11).to_f32(), 0.0);
    }

    #[test]
    fn select_scale_preserves_normal_norm() {
        assert!((select_scale(1.0).to_f32() - 1.0).abs() < FLOAT_EPSILON);
    }

    // -- quantize_coordinates / lookup_centroids roundtrip ---------------------

    #[test]
    fn quantize_lookup_roundtrip_preserves_structure() {
        let codebook = get_codebook(BITS_TQ3, TEST_DIM).unwrap();
        // Use centroid values directly; quantize should map them back.
        let coords: Vec<f32> = codebook.centroids.iter().map(|&c| c as f32).collect();
        let indices = quantize_coordinates(&coords, &codebook);
        let recovered = lookup_centroids(&indices, &codebook);
        // Each recovered value should be exactly the centroid.
        for (i, (&orig, &rec)) in coords.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 0.01,
                "mismatch at index {i}: orig={orig}, rec={rec}"
            );
        }
    }

    // -- PackedBlock construction ---------------------------------------------

    #[test]
    fn packed_block_tq3() {
        let indices = vec![0u8; TEST_DIM];
        let block = PackedBlock::new(BITS_TQ3, f16::from_f32(1.0), &indices);
        assert_eq!(block.bits, BITS_TQ3);
    }

    #[test]
    fn packed_block_tq4() {
        let indices = vec![0u8; TEST_DIM];
        let block = PackedBlock::new(BITS_TQ4, f16::from_f32(1.0), &indices);
        assert_eq!(block.bits, BITS_TQ4);
    }

    // -- quantize rejects dimension mismatch ----------------------------------

    #[test]
    fn quantize_vec_rejects_wrong_dimension() {
        let config = TurboQuantConfig::new(BITS_TQ3, TEST_DIM).unwrap();
        let data = vec![1.0_f32; TEST_DIM + 1];
        assert!(quantize_vec(&config, &data).is_err());
    }

    // -- quantize/dequantize roundtrip ----------------------------------------

    #[test]
    fn quantize_dequantize_roundtrip_3bit() {
        let config = TurboQuantConfig::new(BITS_TQ3, TEST_DIM)
            .unwrap()
            .with_seed(TEST_SEED);
        let data = pseudo_random_vec(TEST_DIM, TEST_SEED_3BIT);
        let block = quantize_vec(&config, &data).unwrap();
        let recovered = dequantize_vec(&config, &block).unwrap();

        let orig_norm = l2_norm(&data);
        let err_norm = l2_norm(
            &data
                .iter()
                .zip(recovered.iter())
                .map(|(&a, &b)| a - b)
                .collect::<Vec<_>>(),
        );
        let relative_error = err_norm / orig_norm;
        // Single-vector relative error can be high; the aggregate MSE check
        // (mse_validation.rs) is the real quality gate.  Here we just verify
        // the pipeline produces a reasonable reconstruction.
        assert!(
            relative_error < MAX_SINGLE_VEC_RELATIVE_ERROR,
            "relative error too large: {relative_error}"
        );
    }

    #[test]
    fn quantize_dequantize_roundtrip_4bit() {
        let config = TurboQuantConfig::new(BITS_TQ4, TEST_DIM)
            .unwrap()
            .with_seed(TEST_SEED);
        let data = pseudo_random_vec(TEST_DIM, TEST_SEED_4BIT);
        let block = quantize_vec(&config, &data).unwrap();
        let recovered = dequantize_vec(&config, &block).unwrap();

        let orig_norm = l2_norm(&data);
        let err_norm = l2_norm(
            &data
                .iter()
                .zip(recovered.iter())
                .map(|(&a, &b)| a - b)
                .collect::<Vec<_>>(),
        );
        let relative_error = err_norm / orig_norm;
        assert!(
            relative_error < MAX_SINGLE_VEC_RELATIVE_ERROR,
            "relative error too large: {relative_error}"
        );
    }

    // -- zero vector ----------------------------------------------------------

    #[test]
    fn quantize_zero_vector_does_not_panic() {
        let config = TurboQuantConfig::new(BITS_TQ3, TEST_DIM)
            .unwrap()
            .with_seed(TEST_SEED);
        let data = vec![0.0_f32; TEST_DIM];
        let block = quantize_vec(&config, &data).unwrap();
        let recovered = dequantize_vec(&config, &block).unwrap();
        let recovered_norm = l2_norm(&recovered);
        assert!(
            recovered_norm < NORM_EPSILON,
            "recovered norm should be near zero, got {recovered_norm}"
        );
    }

    // -- determinism ----------------------------------------------------------

    #[test]
    fn quantize_is_deterministic() {
        let config = TurboQuantConfig::new(BITS_TQ3, TEST_DIM)
            .unwrap()
            .with_seed(TEST_SEED);
        let data = pseudo_random_vec(TEST_DIM, TEST_SEED_DETERM);

        let block_a = quantize_vec(&config, &data).unwrap();
        let block_b = quantize_vec(&config, &data).unwrap();

        let rec_a = dequantize_vec(&config, &block_a).unwrap();
        let rec_b = dequantize_vec(&config, &block_b).unwrap();

        assert_eq!(rec_a, rec_b);
    }

    // -- dequantize_rotated differs from dequantize ---------------------------

    #[test]
    fn dequantize_rotated_differs_from_full() {
        let config = TurboQuantConfig::new(BITS_TQ3, TEST_DIM)
            .unwrap()
            .with_seed(TEST_SEED);
        let data = pseudo_random_vec(TEST_DIM, TEST_SEED_ROTATED);
        let block = quantize_vec(&config, &data).unwrap();

        let full = dequantize_vec(&config, &block).unwrap();
        let rotated = dequantize_rotated(&config, &block).unwrap();

        // They should differ in coordinates...
        assert_ne!(full, rotated);
        // ...but have approximately the same norm (rotation preserves norm).
        let full_norm = l2_norm(&full);
        let rotated_norm = l2_norm(&rotated);
        assert!(
            (full_norm - rotated_norm).abs() < NORM_EPSILON,
            "norms should be approximately equal: full={full_norm}, rotated={rotated_norm}"
        );
    }

    // -- PackedBlock scale and size -------------------------------------------

    #[test]
    fn packed_block_scale_tq3() {
        let block = PackedBlock::new(
            BITS_TQ3,
            f16::from_f32(TEST_CONST_VAL_A),
            &[0u8; TEST_SMALL_DIM],
        );
        assert!((block.scale.to_f32() - TEST_CONST_VAL_A).abs() < 0.01);
    }

    #[test]
    fn packed_block_scale_tq4() {
        let block = PackedBlock::new(
            BITS_TQ4,
            f16::from_f32(TEST_CONST_VAL_B),
            &[0u8; TEST_SMALL_DIM],
        );
        assert!((block.scale.to_f32() - TEST_CONST_VAL_B).abs() < 0.01);
    }

    // -- PackedBlock::size_bytes ----------------------------------------------

    #[test]
    fn packed_block_size_bytes_tq3() {
        let config = TurboQuantConfig::new(BITS_TQ3, TEST_DIM)
            .unwrap()
            .with_seed(TEST_SEED);
        let data = pseudo_random_vec(TEST_DIM, TEST_SEED_3BIT);
        let block = quantize_vec(&config, &data).unwrap();

        // size_bytes = 2 (scale) + packed data length
        assert!(block.size_bytes() > 2);
    }

    #[test]
    fn packed_block_size_bytes_tq4() {
        let config = TurboQuantConfig::new(BITS_TQ4, TEST_DIM)
            .unwrap()
            .with_seed(TEST_SEED);
        let data = pseudo_random_vec(TEST_DIM, TEST_SEED_4BIT);
        let block = quantize_vec(&config, &data).unwrap();

        assert!(block.size_bytes() > 2);
    }

    // -----------------------------------------------------------------------
    // PolarQuant block size verification tests
    // -----------------------------------------------------------------------

    /// Dimension for block-size verification tests.
    const BLOCK_SIZE_DIM: usize = 128;

    /// Bytes per f16 scale field.
    const SCALE_BYTES: usize = 2;

    /// Expected packed size for 2-bit polar, d=128:
    /// packed_indices = 128 / 4 = 32 bytes, + 2 scale = 34 bytes.
    const TQ2_D128_EXPECTED_BYTES: usize = 34;

    /// Expected packed size for 3-bit polar, d=128:
    /// packed_indices = 128 * 3 / 8 = 48 bytes, + 2 scale = 50 bytes.
    const TQ3_D128_EXPECTED_BYTES: usize = 50;

    /// Expected packed size for 4-bit polar, d=128:
    /// packed_indices = 128 / 2 = 64 bytes, + 2 scale = 66 bytes.
    const TQ4_D128_EXPECTED_BYTES: usize = 66;

    /// Seed for block-size verification tests.
    const BLOCK_SIZE_SEED: u64 = 42;

    /// Seed for 2-bit block-size test data.
    const BLOCK_SIZE_DATA_SEED_2: u64 = 20001;

    /// Seed for 3-bit block-size test data.
    const BLOCK_SIZE_DATA_SEED_3: u64 = 30001;

    /// Seed for 4-bit block-size test data.
    const BLOCK_SIZE_DATA_SEED_4: u64 = 40001;

    #[test]
    fn polar_block_size_2bit_d128() {
        let config = TurboQuantConfig::new(BITS_TQ2, BLOCK_SIZE_DIM)
            .unwrap()
            .with_seed(BLOCK_SIZE_SEED);
        let data = pseudo_random_vec(BLOCK_SIZE_DIM, BLOCK_SIZE_DATA_SEED_2);
        let block = quantize_vec(&config, &data).unwrap();

        assert_eq!(
            block.size_bytes(),
            TQ2_D128_EXPECTED_BYTES,
            "2-bit polar block for d={BLOCK_SIZE_DIM}: expected {TQ2_D128_EXPECTED_BYTES} bytes, \
             got {} (scale={SCALE_BYTES}, packed={})",
            block.size_bytes(),
            block.size_bytes() - SCALE_BYTES
        );
    }

    #[test]
    fn polar_block_size_3bit_d128() {
        let config = TurboQuantConfig::new(BITS_TQ3, BLOCK_SIZE_DIM)
            .unwrap()
            .with_seed(BLOCK_SIZE_SEED);
        let data = pseudo_random_vec(BLOCK_SIZE_DIM, BLOCK_SIZE_DATA_SEED_3);
        let block = quantize_vec(&config, &data).unwrap();

        assert_eq!(
            block.size_bytes(),
            TQ3_D128_EXPECTED_BYTES,
            "3-bit polar block for d={BLOCK_SIZE_DIM}: expected {TQ3_D128_EXPECTED_BYTES} bytes, \
             got {} (scale={SCALE_BYTES}, packed={})",
            block.size_bytes(),
            block.size_bytes() - SCALE_BYTES
        );
    }

    #[test]
    fn polar_block_size_4bit_d128() {
        let config = TurboQuantConfig::new(BITS_TQ4, BLOCK_SIZE_DIM)
            .unwrap()
            .with_seed(BLOCK_SIZE_SEED);
        let data = pseudo_random_vec(BLOCK_SIZE_DIM, BLOCK_SIZE_DATA_SEED_4);
        let block = quantize_vec(&config, &data).unwrap();

        assert_eq!(
            block.size_bytes(),
            TQ4_D128_EXPECTED_BYTES,
            "4-bit polar block for d={BLOCK_SIZE_DIM}: expected {TQ4_D128_EXPECTED_BYTES} bytes, \
             got {} (scale={SCALE_BYTES}, packed={})",
            block.size_bytes(),
            block.size_bytes() - SCALE_BYTES
        );
    }
}
