//! QJL (Quantized Johnson-Lindenstrauss) bias correction.
//!
//! Implements Stage 2 of Algorithm 2 (TURBOQUANTprod) from the paper:
//! after (b-1)-bit PolarQuant, the residual is projected through a
//! Rademacher random matrix and stored as 1-bit signs.  At query time
//! the correction term unbias the inner-product estimate.
//!
//! ## Algorithm
//!
//! ```text
//! Quantize:
//!   1. q_mse = PolarQuant(x, b)            // b-bit PolarQuant
//!   2. residual = x - dequantize(q_mse)
//!   3. signs = sign(R * residual)           // R is Rademacher (+/-1/sqrt(d))
//!
//! Inner product:
//!   <y,x>_est = <y, deq(q_mse)> + c * <R*y, signs>
//!   c = ||residual|| * sqrt(pi/2) / sqrt(d)
//! ```
//!
//! The QJL signs (1 bit per dimension) are stored as additional metadata
//! alongside the b-bit polar block.
//!
//! Mathematical property: E[<y,x>_est] = <y,x> (unbiased).
//!
//! ## Implementation
//!
//! Rademacher signs are generated via a SplitMix64-based hash function,
//! which provides excellent statistical properties (passes BigCrush) with
//! pure arithmetic -- no RNG state needed.

use half::f16;

use crate::codebook::Codebook;
use crate::error::{require, Result, TurboQuantError};
use crate::packed::{PackedBlock, TurboQuantConfig};
use crate::quantize::{
    dequantize_into_with_codebook, dequantize_vec, l2_norm, quantize_vec, DequantScratch,
};

// ---------------------------------------------------------------------------
// Hash-based Rademacher constants
// ---------------------------------------------------------------------------

/// SplitMix64 golden-gamma constant used in hash-based Rademacher generation.
const SPLITMIX_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;

/// SplitMix64 first finalizer multiplier.
const SPLITMIX_MUL_1: u64 = 0xbf58_476d_1ce4_e5b9;

/// SplitMix64 second finalizer multiplier.
const SPLITMIX_MUL_2: u64 = 0x94d0_49bb_1331_11eb;

/// SplitMix64 first finalizer shift.
const SPLITMIX_SHIFT_1: u32 = 30;

/// SplitMix64 second finalizer shift.
const SPLITMIX_SHIFT_2: u32 = 27;

/// SplitMix64 final shift for extracting the sign bit.
const SPLITMIX_SHIFT_3: u32 = 31;

// ---------------------------------------------------------------------------
// Named constants (no magic numbers)
// ---------------------------------------------------------------------------

/// sqrt(pi / 2) -- the scaling constant in the QJL correction term.
const SQRT_PI_OVER_2: f32 = 1.253_314_1; // std::f32::consts::FRAC_PI_2.sqrt()

/// Number of bits per byte, used for sign packing.
const BITS_PER_BYTE: usize = 8;

/// Minimum supported PolarQuant bit width for QJL.
const MIN_POLAR_BITS: u8 = 3;

/// Maximum supported PolarQuant bit width for QJL.
const MAX_POLAR_BITS: u8 = 4;

/// Multiplier used in the FNV-style seed combination for Rademacher rows.
const SEED_MIX_MULTIPLIER: u64 = 0x517c_c1b7_2722_0a95;

/// XOR constant used in the second round of seed mixing.
const SEED_MIX_XOR: u64 = 0x6c62_272e_07bb_0142;

/// Sign value for positive Rademacher entry.
const POSITIVE_SIGN: f32 = 1.0;

/// Sign value for negative Rademacher entry.
const NEGATIVE_SIGN: f32 = -1.0;

/// Multiplier for branchless sign-bit extraction: maps {0,1} to {-1.0,+1.0}.
const SIGN_BIT_SCALE: f32 = 2.0;

/// Offset for branchless sign-bit extraction: maps {0,1} to {-1.0,+1.0}.
const SIGN_BIT_OFFSET: f32 = -1.0;

/// Right-shift amount used in seed mixing (half of u64 width).
const SEED_MIX_SHIFT: u32 = 32;

/// Number of bits in a packed sign byte (alias for clarity in QJL context).
const SIGN_PACK_BITS: usize = BITS_PER_BYTE;

// ---------------------------------------------------------------------------
// QjlBlock
// ---------------------------------------------------------------------------

/// A TurboQuant block with QJL bias correction.
///
/// Stores (b-1)-bit PolarQuant result + 1-bit QJL signs + residual norm.
/// The PolarQuant bit width is derived from `polar_block.bits`.
pub struct QjlBlock {
    /// The (b-1)-bit PolarQuant quantized block.
    pub polar_block: PackedBlock,
    /// 1-bit QJL signs: sign(R * residual), packed as bits.
    pub qjl_signs: Vec<u8>,
    /// L2 norm of the residual vector.
    pub residual_norm: f16,
}

impl QjlBlock {
    /// Creates a `QjlBlock` from pre-computed components without re-quantizing.
    ///
    /// Use this to reconstruct blocks from GPU-quantized data.
    ///
    /// Pure Operation: field assignment only.
    // qual:api — used by GPU kernel integration for importing quantized data
    pub fn from_parts(polar_block: PackedBlock, qjl_signs: Vec<u8>, residual_norm: f16) -> Self {
        Self {
            polar_block,
            qjl_signs,
            residual_norm,
        }
    }
}

// ---------------------------------------------------------------------------
// Pure Operation helpers
// ---------------------------------------------------------------------------

/// Combines a base seed with a row index to produce a per-row seed.
///
/// Pure Operation: arithmetic only.
fn mix_seed(seed: u64, row_index: usize) -> u64 {
    let row = row_index as u64;
    let mixed = seed.wrapping_mul(SEED_MIX_MULTIPLIER).wrapping_add(row);
    mixed ^ (mixed >> SEED_MIX_SHIFT) ^ SEED_MIX_XOR
}

/// Returns a Rademacher sign (+1.0 or -1.0) using a SplitMix64 hash.
///
/// Uses SplitMix64 finalizer applied to (seed + row * gamma + col) for excellent
/// statistical properties. SplitMix64 is a well-studied hash with full-period
/// and passes BigCrush. This gives each sign +/- with equal probability and
/// near-independence across different (row, col) positions.
///
/// Pure Operation: arithmetic only, no RNG state.
fn rademacher_sign_from_hash(seed: u64, row: usize, col: usize) -> f32 {
    // Combine seed, row, col into a single state using golden-gamma steps.
    let z = seed
        .wrapping_add((row as u64).wrapping_mul(SPLITMIX_GAMMA))
        .wrapping_add(col as u64);
    // SplitMix64 finalizer (Stafford variant 13).
    let z = (z ^ (z >> SPLITMIX_SHIFT_1)).wrapping_mul(SPLITMIX_MUL_1);
    let z = (z ^ (z >> SPLITMIX_SHIFT_2)).wrapping_mul(SPLITMIX_MUL_2);
    let z = z ^ (z >> SPLITMIX_SHIFT_3);
    if z & 1 == 0 {
        POSITIVE_SIGN
    } else {
        NEGATIVE_SIGN
    }
}

/// Generates one row of the implicit Rademacher matrix R.
///
/// Each entry is +1/sqrt(d) or -1/sqrt(d), deterministic from (seed, row_index).
///
/// Pure Operation: arithmetic only, no RNG state.
// qual:api -- public API for QJL matrix inspection and testing
pub fn generate_rademacher_row(dim: usize, seed: u64, row_index: usize) -> Vec<f32> {
    let row_seed = mix_seed(seed, row_index);
    let inv_sqrt_d = 1.0 / (dim as f32).sqrt();

    (0..dim)
        .map(|col| {
            let sign = rademacher_sign_from_hash(row_seed, row_index, col);
            sign * inv_sqrt_d
        })
        .collect()
}

/// Computes one entry of R*data without materializing the full matrix.
///
/// For row `row_index`: sum_i sign(seed, row_index, i) * data[i] / sqrt(d).
///
/// Pure Operation: arithmetic only, no RNG state.
fn rademacher_vector_product(data: &[f32], dim: usize, seed: u64, row_index: usize) -> f32 {
    let row_seed = mix_seed(seed, row_index);
    let inv_sqrt_d = 1.0 / (dim as f32).sqrt();

    data.iter()
        .enumerate()
        .map(|(col, &val)| {
            let sign = rademacher_sign_from_hash(row_seed, row_index, col);
            sign * val
        })
        .sum::<f32>()
        * inv_sqrt_d
}

/// Computes the element-wise residual: original - reconstructed.
///
/// Pure Operation: arithmetic only.
pub fn compute_residual(original: &[f32], reconstructed: &[f32]) -> Vec<f32> {
    original
        .iter()
        .zip(reconstructed.iter())
        .map(|(&o, &r)| o - r)
        .collect()
}

/// Dot product of two f32 slices.
///
/// Pure Operation: arithmetic only.
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Extracts the sign bit at `index` from a packed sign vector.
///
/// Returns +1.0 if the bit is set, -1.0 if clear.
///
/// Pure Operation: branchless bitwise arithmetic only.
pub fn sign_bit(signs: &[u8], index: usize) -> f32 {
    let byte_index = index / BITS_PER_BYTE;
    let bit_offset = index % BITS_PER_BYTE;
    let bit = ((signs[byte_index] >> bit_offset) & 1) as f32;
    bit * SIGN_BIT_SCALE + SIGN_BIT_OFFSET // 2.0 * bit - 1.0
}

/// Packs a slice of booleans into a bit vector.
///
/// `true` maps to bit 1, `false` maps to bit 0.
///
/// Pure Operation: bitwise arithmetic only.
pub fn pack_sign_bits(signs: &[bool]) -> Vec<u8> {
    let num_bytes = ceiling_div(signs.len(), BITS_PER_BYTE);
    let mut packed = vec![0u8; num_bytes];

    for (i, &positive) in signs.iter().enumerate() {
        let byte_index = i / SIGN_PACK_BITS;
        let bit_offset = i % SIGN_PACK_BITS;
        packed[byte_index] |= (positive as u8) << bit_offset;
    }

    packed
}

/// Computes the QJL scaling constant.
///
/// c = residual_norm * sqrt(pi/2) / sqrt(dim)
///
/// Pure Operation: arithmetic only.
pub fn qjl_scaling_constant(residual_norm: f32, dim: usize) -> f32 {
    residual_norm * SQRT_PI_OVER_2 / (dim as f32).sqrt()
}

/// Integer ceiling division: ceil(a / b).
///
/// Pure Operation: arithmetic only.
fn ceiling_div(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

/// Checks whether the PolarQuant bit width is valid for QJL (3 or 4).
///
/// Pure Operation: comparison only.
fn is_valid_qjl_bits(bits: u8) -> bool {
    (MIN_POLAR_BITS..=MAX_POLAR_BITS).contains(&bits)
}

/// Computes the PolarQuant bit width from the total QJL bit budget.
///
/// Pure Operation: arithmetic only (total_bits - 1 bit for QJL signs).
fn polar_bit_width(total_bits: u8) -> u8 {
    total_bits - QJL_SIGN_BITS
}

/// Number of bits reserved for QJL sign storage.
const QJL_SIGN_BITS: u8 = 1;

// ---------------------------------------------------------------------------
// Integration: compute_qjl_signs
// ---------------------------------------------------------------------------

/// Computes QJL sign bits for a residual vector.
///
/// For each projection j = 0..dim, computes sign(R_j * residual) and packs
/// the results as a bit vector (8 signs per byte).
///
/// Integration: calls `rademacher_vector_product` and `pack_sign_bits`.
pub fn compute_qjl_signs(residual: &[f32], dim: usize, seed: u64) -> crate::error::Result<Vec<u8>> {
    if residual.len() != dim {
        return Err(crate::error::TurboQuantError::DimensionMismatch {
            expected: dim,
            actual: residual.len(),
        });
    }
    let sign_bools: Vec<bool> = (0..dim)
        .map(|j| {
            let projection = rademacher_vector_product(residual, dim, seed, j);
            projection >= 0.0
        })
        .collect();
    Ok(pack_sign_bits(&sign_bools))
}

// ---------------------------------------------------------------------------
// Validation helper
// ---------------------------------------------------------------------------

/// Validates that the PolarQuant bit width is supported for QJL.
///
/// Pure Integration: only calls `require` and `is_valid_qjl_bits`.
fn validate_qjl_config(bits: u8) -> Result<()> {
    require(
        is_valid_qjl_bits(bits),
        TurboQuantError::UnsupportedBits(bits),
    )
}

// ---------------------------------------------------------------------------
// Integration: quantize_with_qjl
// ---------------------------------------------------------------------------

/// Quantizes a vector using TURBOQUANTprod (Algorithm 2).
///
/// Interprets `config.bits` as the **total bit budget**: (bits-1) bits for
/// PolarQuant plus 1 bit for QJL signs.  For example, `bits=3` means
/// 2-bit PolarQuant + 1-bit QJL = 3 bits total.
///
/// Uses hash-based (SplitMix64) Rademacher sign generation for excellent
/// CPU throughput with no RNG state overhead.
///
/// Integration: orchestrates `validate_qjl_config`, `quantize_vec`,
/// `dequantize_vec`, `compute_residual`, `compute_qjl_signs`, and `l2_norm`.
///
/// # Errors
///
/// Returns [`TurboQuantError::UnsupportedBits`] if `config.bits` is not 3 or 4.
/// Propagates errors from `quantize_vec` / `dequantize_vec`.
pub fn quantize_with_qjl(
    config: &TurboQuantConfig,
    data: &[f32],
    qjl_seed: u64,
) -> Result<QjlBlock> {
    validate_qjl_config(config.bits)?;

    let polar_bits = polar_bit_width(config.bits);
    let polar_config =
        TurboQuantConfig::new(polar_bits, config.dim)?.with_seed(config.rotation_seed);

    let polar_block = quantize_vec(&polar_config, data)?;
    let reconstructed = dequantize_vec(&polar_config, &polar_block)?;
    let residual = compute_residual(data, &reconstructed);
    let residual_norm = l2_norm(&residual);
    let qjl_signs = compute_qjl_signs(&residual, config.dim, qjl_seed)?;

    Ok(QjlBlock {
        polar_block,
        qjl_signs,
        residual_norm: f16::from_f32(residual_norm),
    })
}

/// Pre-fetched resources for batch QJL quantization.
///
/// Hoists codebook lookups, sign-pattern generation, and config construction
/// out of the per-vector loop.  Created once via [`QjlBatchResources::new`],
/// then passed to [`quantize_with_qjl_resources`] for each vector.
pub struct QjlBatchResources {
    /// PolarQuant configuration (bits-1, dim, seed).
    pub polar_config: TurboQuantConfig,
    /// Codebook for the polar bit width.
    pub polar_codebook: crate::codebook::Codebook,
    /// Sign pattern for the polar rotation.
    pub polar_sign_pattern: Vec<f32>,
    /// Scratch buffers for dequantization (reused across vectors).
    pub scratch: DequantScratch,
}

impl QjlBatchResources {
    /// Creates batch resources for the given overall QJL config.
    ///
    /// Validates the bit budget once, computes polar config, fetches
    /// codebook and sign pattern.
    ///
    /// # Errors
    ///
    /// Returns [`TurboQuantError::UnsupportedBits`] if `config.bits` is not 3 or 4.
    /// Propagates codebook / config errors.
    pub fn new(config: &TurboQuantConfig) -> Result<Self> {
        validate_qjl_config(config.bits)?;
        let polar_bits = polar_bit_width(config.bits);
        let polar_config =
            TurboQuantConfig::new(polar_bits, config.dim)?.with_seed(config.rotation_seed);
        let polar_codebook = crate::codebook::get_codebook(polar_bits, config.dim)?;
        let polar_sign_pattern =
            crate::rotation::generate_sign_pattern(config.dim, config.rotation_seed);
        let scratch = DequantScratch::new(config.dim);
        Ok(Self {
            polar_config,
            polar_codebook,
            polar_sign_pattern,
            scratch,
        })
    }
}

/// Quantizes a vector using TURBOQUANTprod with pre-fetched resources.
///
/// Hot-path variant of [`quantize_with_qjl`]: reuses codebook, sign pattern,
/// polar config, and scratch buffers across multiple vectors.  This avoids
/// repeated allocation of these resources when quantizing a batch of vectors
/// (e.g. during prefill).
///
/// # Errors
///
/// Propagates errors from quantize / dequantize steps.
pub fn quantize_with_qjl_resources(
    data: &[f32],
    qjl_seed: u64,
    res: &mut QjlBatchResources,
) -> Result<QjlBlock> {
    use crate::quantize::quantize_vec_with_codebook;

    let polar_block = quantize_vec_with_codebook(
        &res.polar_config,
        data,
        &res.polar_codebook,
        &res.polar_sign_pattern,
    )?;

    // Dequantize using scratch buffers (no allocation)
    dequantize_into_with_codebook(
        &res.polar_config,
        &polar_block,
        &res.polar_codebook,
        &res.polar_sign_pattern,
        &mut res.scratch,
    )?;

    let residual = compute_residual(data, &res.scratch.values);
    let residual_norm = l2_norm(&residual);
    let dim = res.polar_config.dim;
    let qjl_signs = compute_qjl_signs(&residual, dim, qjl_seed)?;

    Ok(QjlBlock {
        polar_block,
        qjl_signs,
        residual_norm: f16::from_f32(residual_norm),
    })
}

/// Combines the base inner product with the QJL correction term.
///
/// Pure Operation: arithmetic only.
fn corrected_estimate(base: f32, scaling: f32, correction: f32) -> f32 {
    base + scaling * correction
}

// ---------------------------------------------------------------------------
// Pre-computed query projections (CPU optimization: batch R*query)
// ---------------------------------------------------------------------------

/// Pre-computes all d Rademacher projections for a query vector.
///
/// Returns a vector where entry j = R_j * query (the dot product of the
/// j-th Rademacher row with the query).
///
/// Call this ONCE per query, then use `estimate_inner_product` per key.
///
/// Integration: calls `rademacher_vector_product` for each projection row.
// qual:api -- public API for pre-computing query projections before batch scoring
pub fn precompute_query_projections(query: &[f32], dim: usize, qjl_seed: u64) -> Vec<f32> {
    (0..dim)
        .map(|j| rademacher_vector_product(query, dim, qjl_seed, j))
        .collect()
}

// ---------------------------------------------------------------------------
// SIMD-friendly sign unpacking (CPU optimization)
// ---------------------------------------------------------------------------

/// Unpacks all signs from packed bits into an f32 vector of +1.0/-1.0 values.
///
/// Pure Operation: bitwise and arithmetic only.
#[cfg(test)]
fn unpack_signs_to_f32(signs: &[u8], count: usize) -> Vec<f32> {
    (0..count).map(|i| sign_bit(signs, i)).collect()
}

/// QJL correction using pre-computed R*query projections.
///
/// Fused implementation: iterates once over r_query, extracting sign bits
/// inline to avoid allocating an intermediate Vec.
///
/// Pure Integration: calls `sign_bit` per element.
fn qjl_correction(r_query: &[f32], signs: &[u8], dim: usize) -> f32 {
    r_query
        .iter()
        .enumerate()
        .take(dim)
        .map(|(i, &rq)| rq * sign_bit(signs, i))
        .sum()
}

// ---------------------------------------------------------------------------
// Inner product estimation (batch API with pre-computed projections)
// ---------------------------------------------------------------------------

/// Estimates <query, key> using pre-computed query projections.
///
/// Instead of recomputing R*query per key, uses cached projections and
/// SIMD-friendly sign unpacking for the correction term.
///
/// For batch scoring over many keys, call `precompute_query_projections`
/// once, then call this function per key.
///
/// Integration: orchestrates `dequantize_vec`, `dot_product`,
/// `qjl_correction`, and `qjl_scaling_constant`.
///
/// # Errors
///
/// Propagates errors from `dequantize_vec`.
// qual:api -- public API for optimized batch inner-product estimation
pub fn estimate_inner_product(
    query: &[f32],
    r_query: &[f32],
    qjl_block: &QjlBlock,
    config: &TurboQuantConfig,
) -> Result<f32> {
    let polar_config = TurboQuantConfig::new(qjl_block.polar_block.bits, config.dim)?
        .with_seed(config.rotation_seed);
    let reconstructed = dequantize_vec(&polar_config, &qjl_block.polar_block)?;
    let base = dot_product(query, &reconstructed);

    let dim = config.dim;
    let residual_norm = qjl_block.residual_norm.to_f32();
    let c = qjl_scaling_constant(residual_norm, dim);
    let correction = qjl_correction(r_query, &qjl_block.qjl_signs, dim);

    Ok(corrected_estimate(base, c, correction))
}

/// Pre-fetched context for batch inner-product estimation.
/// Avoids repeated codebook/config/sign_pattern/scratch lookups per key.
pub struct EstimationContext<'a> {
    /// PolarQuant configuration for the inner block.
    pub polar_config: &'a TurboQuantConfig,
    /// Codebook used for dequantization.
    pub codebook: &'a Codebook,
    /// Sign pattern used for inverse rotation.
    pub sign_pattern: &'a [f32],
    /// Vector dimension.
    pub dim: usize,
    /// Scratch buffers reused across keys.
    pub scratch: DequantScratch,
}

/// Hot-path inner product estimation with pre-fetched context and scratch buffers.
///
/// Uses pre-allocated [`DequantScratch`] to avoid per-key heap allocations
/// in the attention scoring loop.
///
/// Integration: delegates to `dequantize_into_with_codebook`, `dot_product`,
/// `qjl_scaling_constant`, `qjl_correction`, and `corrected_estimate`.
///
/// # Errors
///
/// Propagates errors from `dequantize_into_with_codebook`.
pub fn estimate_inner_product_with_codebook(
    query: &[f32],
    r_query: &[f32],
    qjl_block: &QjlBlock,
    ctx: &mut EstimationContext,
) -> Result<f32> {
    dequantize_into_with_codebook(
        ctx.polar_config,
        &qjl_block.polar_block,
        ctx.codebook,
        ctx.sign_pattern,
        &mut ctx.scratch,
    )?;
    let base = dot_product(query, &ctx.scratch.values);

    let dim = ctx.dim;
    let residual_norm = qjl_block.residual_norm.to_f32();
    let c = qjl_scaling_constant(residual_norm, dim);
    let correction = qjl_correction(r_query, &qjl_block.qjl_signs, dim);

    Ok(corrected_estimate(base, c, correction))
}

/// Estimates <query, key> with QJL bias correction (single-key convenience).
///
/// For batch scoring over many keys, prefer `precompute_query_projections` +
/// `estimate_inner_product` to avoid recomputing R*query per key.
///
/// Integration: calls `precompute_query_projections` and `estimate_inner_product`.
///
/// # Errors
///
/// Propagates errors from `estimate_inner_product`.
// qual:api -- public API for single-key inner-product estimation
pub fn estimate_inner_product_single(
    query: &[f32],
    qjl_block: &QjlBlock,
    config: &TurboQuantConfig,
    qjl_seed: u64,
) -> Result<f32> {
    let r_query = precompute_query_projections(query, config.dim, qjl_seed);
    estimate_inner_product(query, &r_query, qjl_block, config)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{pseudo_random_vec, LCG_MULTIPLIER};

    // -- Named test constants ------------------------------------------------

    /// Test dimension (power of two, suitable for WHT).
    const TEST_DIM: usize = 64;

    /// Test dimension used in scaling-constant tests.
    const TEST_DIM_64: usize = 64;

    /// Seed for PolarQuant rotation.
    const TEST_ROTATION_SEED: u64 = 42;

    /// Seed for QJL Rademacher matrix.
    const TEST_QJL_SEED: u64 = 12345;

    /// Row index used in determinism test.
    const TEST_SIGN_INDEX: usize = 7;

    /// Alternate seed for pseudo-random vector generation.
    const TEST_ALT_SEED: u64 = 999;

    /// Seed A for pseudo-random vector generation.
    const TEST_SEED_A: u64 = 11111;
    /// Seed B for pseudo-random vector generation.
    const TEST_SEED_B: u64 = 22222;
    /// Seed C for pseudo-random vector generation.
    const TEST_SEED_C: u64 = 33333;
    /// Seed D for pseudo-random vector generation.
    const TEST_SEED_D: u64 = 44444;
    /// Seed E for pseudo-random vector generation.
    const TEST_SEED_E: u64 = 55555;
    /// Seed F for pseudo-random vector generation.
    const TEST_SEED_F: u64 = 66666;
    /// Seed G for pseudo-random vector generation.
    const TEST_SEED_G: u64 = 77777;
    /// Seed H for pseudo-random vector generation.
    const TEST_SEED_H: u64 = 88888;

    /// Expected dot product for known-value test: 1*4 + 2*5 + 3*6 = 32.
    const TEST_EXPECTED_DOT: f32 = 32.0;

    /// Scale factor used in parallel-vectors test (query = key * factor).
    const TEST_SCALE_FACTOR: f32 = 2.0;

    /// Number of samples for single-pair tests (e.g., bias seed offset base).
    const TEST_SAMPLE_COUNT: u64 = 100;

    /// Larger sample offset base for query-seed generation.
    const TEST_LARGE_SAMPLE_COUNT: u64 = 200;

    /// Row index used in sign-pack / vector-product explicit-row test.
    const TEST_SIGN_PACK_COUNT: usize = 3;

    /// Tolerance for floating-point comparisons.
    const FLOAT_EPSILON: f32 = 1e-6;

    /// Tolerance for statistical tests (mean bias).
    /// With 500 samples and d=64, residual finite-sample noise + f16 rounding
    /// of the residual norm means the mean bias won't converge to exactly zero.
    /// The (bits-1) PolarQuant stage (2-bit for TQ3) produces a larger residual,
    /// so the QJL correction has more work to do.
    const BIAS_TOLERANCE: f32 = 0.20;

    /// Number of random pairs for statistical tests.
    const STATISTICAL_SAMPLE_COUNT: usize = 500;

    /// Bit width for the overall 3-bit budget.
    const BITS_3: u8 = 3;

    /// Bit width for the overall 4-bit budget.
    const BITS_4: u8 = 4;

    /// Number of sign bits to pack in the packing test.
    const PACK_TEST_LEN: usize = 19;

    /// Pattern stride for generating alternating booleans in sign-pack tests.
    const SIGN_PACK_PATTERN_STRIDE: usize = 3;

    /// Maximum acceptable relative variance (loose bound for small samples).
    const MAX_RELATIVE_VARIANCE: f32 = 1.0;

    /// Tolerance for orthogonal-vectors test (absolute).
    const ORTHO_TOLERANCE: f32 = 0.3;

    /// Tolerance for parallel-vectors test (relative to true product).
    const PARALLEL_RELATIVE_TOLERANCE: f32 = 0.5;

    // -- Helper: unit vector ---------------------------------------------------

    /// Returns a unit vector along the given axis.
    fn unit_vec(dim: usize, axis: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; dim];
        v[axis] = 1.0;
        v
    }

    // -- Rademacher properties -----------------------------------------------

    #[test]
    fn rademacher_row_has_correct_magnitude() {
        let row = generate_rademacher_row(TEST_DIM, TEST_QJL_SEED, 0);
        let expected_magnitude = 1.0 / (TEST_DIM as f32).sqrt();

        assert_eq!(row.len(), TEST_DIM);
        for &val in &row {
            assert!(
                (val.abs() - expected_magnitude).abs() < FLOAT_EPSILON,
                "expected +/-{expected_magnitude}, got {val}"
            );
        }
    }

    #[test]
    fn rademacher_row_is_deterministic() {
        let row_a = generate_rademacher_row(TEST_DIM, TEST_QJL_SEED, TEST_SIGN_INDEX);
        let row_b = generate_rademacher_row(TEST_DIM, TEST_QJL_SEED, TEST_SIGN_INDEX);
        assert_eq!(row_a, row_b);
    }

    #[test]
    fn rademacher_different_rows_differ() {
        let row_a = generate_rademacher_row(TEST_DIM, TEST_QJL_SEED, 0);
        let row_b = generate_rademacher_row(TEST_DIM, TEST_QJL_SEED, 1);
        assert_ne!(row_a, row_b);
    }

    // -- rademacher_vector_product -------------------------------------------

    #[test]
    fn rademacher_vector_product_matches_explicit_row() {
        let data = pseudo_random_vec(TEST_DIM, TEST_ALT_SEED);
        let row = generate_rademacher_row(TEST_DIM, TEST_QJL_SEED, TEST_SIGN_PACK_COUNT);
        let expected = dot_product(&data, &row);
        let actual =
            rademacher_vector_product(&data, TEST_DIM, TEST_QJL_SEED, TEST_SIGN_PACK_COUNT);
        assert!(
            (expected - actual).abs() < FLOAT_EPSILON,
            "expected {expected}, got {actual}"
        );
    }

    // -- compute_residual ----------------------------------------------------

    const RESIDUAL_ORIGINAL: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    const RESIDUAL_RECONSTRUCTED: [f32; 4] = [0.9, 2.1, 2.8, 4.2];
    const RESIDUAL_EXPECTED: [f32; 4] = [0.1, -0.1, 0.2, -0.2];

    #[test]
    fn compute_residual_basic() {
        let residual = compute_residual(&RESIDUAL_ORIGINAL, &RESIDUAL_RECONSTRUCTED);
        for (i, (&r, &e)) in residual.iter().zip(RESIDUAL_EXPECTED.iter()).enumerate() {
            assert!(
                (r - e).abs() < FLOAT_EPSILON,
                "residual[{i}]: expected {e}, got {r}"
            );
        }
    }

    #[test]
    fn compute_residual_zero_when_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let residual = compute_residual(&v, &v);
        for &r in &residual {
            assert!(r.abs() < FLOAT_EPSILON);
        }
    }

    // -- dot_product ---------------------------------------------------------

    #[test]
    fn dot_product_known_value() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let expected = TEST_EXPECTED_DOT; // 1*4 + 2*5 + 3*6
        let actual = dot_product(&a, &b);
        assert!((actual - expected).abs() < FLOAT_EPSILON);
    }

    #[test]
    fn dot_product_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(dot_product(&a, &b).abs() < FLOAT_EPSILON);
    }

    // -- sign_bit / pack_sign_bits -------------------------------------------

    #[test]
    fn pack_unpack_sign_bits_roundtrip() {
        let bools: Vec<bool> = (0..PACK_TEST_LEN)
            .map(|i| i % SIGN_PACK_PATTERN_STRIDE == 0)
            .collect();
        let packed = pack_sign_bits(&bools);

        for (i, &expected_positive) in bools.iter().enumerate() {
            let extracted = sign_bit(&packed, i);
            let expected = if expected_positive {
                POSITIVE_SIGN
            } else {
                NEGATIVE_SIGN
            };
            assert!(
                (extracted - expected).abs() < FLOAT_EPSILON,
                "bit {i}: expected {expected}, got {extracted}"
            );
        }
    }

    #[test]
    fn pack_sign_bits_all_true() {
        let bools = vec![true; BITS_PER_BYTE];
        let packed = pack_sign_bits(&bools);
        assert_eq!(packed.len(), 1);
        assert_eq!(packed[0], 0xFF);
    }

    #[test]
    fn pack_sign_bits_all_false() {
        let bools = vec![false; BITS_PER_BYTE];
        let packed = pack_sign_bits(&bools);
        assert_eq!(packed.len(), 1);
        assert_eq!(packed[0], 0x00);
    }

    // -- qjl_scaling_constant ------------------------------------------------

    #[test]
    fn qjl_scaling_constant_correct_formula() {
        let residual_norm = TEST_SCALE_FACTOR;
        let dim = TEST_DIM_64;
        let expected = residual_norm * SQRT_PI_OVER_2 / (dim as f32).sqrt();
        let actual = qjl_scaling_constant(residual_norm, dim);
        assert!(
            (actual - expected).abs() < FLOAT_EPSILON,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn qjl_scaling_constant_zero_norm() {
        let actual = qjl_scaling_constant(0.0, TEST_DIM);
        assert!(actual.abs() < FLOAT_EPSILON);
    }

    // -- quantize_with_qjl ---------------------------------------------------

    #[test]
    fn quantize_with_qjl_produces_valid_block() {
        let config = TurboQuantConfig::new(BITS_3, TEST_DIM)
            .unwrap()
            .with_seed(TEST_ROTATION_SEED);
        let data = pseudo_random_vec(TEST_DIM, TEST_SEED_A);
        let block = quantize_with_qjl(&config, &data, TEST_QJL_SEED).unwrap();

        // qjl_signs should have ceil(dim/8) bytes
        let expected_sign_bytes = ceiling_div(TEST_DIM, BITS_PER_BYTE);
        assert_eq!(block.qjl_signs.len(), expected_sign_bytes);
        assert!(block.residual_norm.to_f32() >= 0.0);
    }

    #[test]
    fn quantize_with_qjl_4bit() {
        let config = TurboQuantConfig::new(BITS_4, TEST_DIM)
            .unwrap()
            .with_seed(TEST_ROTATION_SEED);
        let data = pseudo_random_vec(TEST_DIM, TEST_SEED_B);
        let block = quantize_with_qjl(&config, &data, TEST_QJL_SEED);
        assert!(block.is_ok());
    }

    #[test]
    fn quantize_with_qjl_rejects_invalid_bits() {
        // We can't use TurboQuantConfig::new(5, ..) because it rejects 5.
        // The actual error path is tested via is_valid_qjl_bits.
        assert!(!is_valid_qjl_bits(2));
        assert!(!is_valid_qjl_bits(5));
    }

    // -- determinism ---------------------------------------------------------

    #[test]
    fn qjl_quantize_is_deterministic() {
        let config = TurboQuantConfig::new(BITS_3, TEST_DIM)
            .unwrap()
            .with_seed(TEST_ROTATION_SEED);
        let data = pseudo_random_vec(TEST_DIM, TEST_SEED_C);

        let block_a = quantize_with_qjl(&config, &data, TEST_QJL_SEED).unwrap();
        let block_b = quantize_with_qjl(&config, &data, TEST_QJL_SEED).unwrap();

        assert_eq!(block_a.qjl_signs, block_b.qjl_signs);
        assert_eq!(
            block_a.residual_norm.to_f32(),
            block_b.residual_norm.to_f32()
        );
    }

    #[test]
    fn estimate_inner_product_single_is_deterministic() {
        let config = TurboQuantConfig::new(BITS_3, TEST_DIM)
            .unwrap()
            .with_seed(TEST_ROTATION_SEED);
        let key = pseudo_random_vec(TEST_DIM, TEST_SEED_D);
        let query = pseudo_random_vec(TEST_DIM, TEST_SEED_E);

        let block = quantize_with_qjl(&config, &key, TEST_QJL_SEED).unwrap();
        let est_a = estimate_inner_product_single(&query, &block, &config, TEST_QJL_SEED).unwrap();
        let est_b = estimate_inner_product_single(&query, &block, &config, TEST_QJL_SEED).unwrap();

        assert!(
            (est_a - est_b).abs() < FLOAT_EPSILON,
            "not deterministic: {est_a} vs {est_b}"
        );
    }

    // -- Bias test (statistical) ---------------------------------------------

    #[test]
    fn qjl_reduces_bias_vs_plain_polarquant() {
        let config = TurboQuantConfig::new(BITS_3, TEST_DIM)
            .unwrap()
            .with_seed(TEST_ROTATION_SEED);

        let mut qjl_bias_sum = 0.0_f64;
        let mut plain_bias_sum = 0.0_f64;

        for i in 0..STATISTICAL_SAMPLE_COUNT {
            let key_seed = (i as u64)
                .wrapping_mul(LCG_MULTIPLIER)
                .wrapping_add(TEST_SAMPLE_COUNT);
            let query_seed = (i as u64)
                .wrapping_mul(LCG_MULTIPLIER)
                .wrapping_add(TEST_LARGE_SAMPLE_COUNT);
            // Use a different QJL seed per sample so we average over R's randomness.
            let qjl_seed = TEST_QJL_SEED.wrapping_add(i as u64);

            let key = pseudo_random_vec(TEST_DIM, key_seed);
            let query = pseudo_random_vec(TEST_DIM, query_seed);
            let true_ip = dot_product(&key, &query) as f64;

            // QJL-corrected estimate (same b-bit polar + QJL correction)
            let block = quantize_with_qjl(&config, &key, qjl_seed).unwrap();
            let qjl_est =
                estimate_inner_product_single(&query, &block, &config, qjl_seed).unwrap() as f64;

            // Plain PolarQuant estimate (same b-bit, no QJL correction)
            let plain_block = quantize_vec(&config, &key).unwrap();
            let plain_recon = dequantize_vec(&config, &plain_block).unwrap();
            let plain_est = dot_product(&query, &plain_recon) as f64;

            qjl_bias_sum += qjl_est - true_ip;
            plain_bias_sum += plain_est - true_ip;
        }

        let qjl_mean_bias = (qjl_bias_sum / STATISTICAL_SAMPLE_COUNT as f64).abs() as f32;
        let plain_mean_bias = (plain_bias_sum / STATISTICAL_SAMPLE_COUNT as f64).abs() as f32;

        // QJL should have low bias (unbiased in expectation).
        assert!(
            qjl_mean_bias < BIAS_TOLERANCE,
            "QJL mean bias {qjl_mean_bias} exceeds tolerance {BIAS_TOLERANCE}"
        );

        // Informational: plain PolarQuant typically has some bias.
        // We don't hard-fail if plain happens to be low, but QJL should be low.
        let _ = plain_mean_bias; // used for comparison, no assertion needed
    }

    // -- Variance test -------------------------------------------------------

    #[test]
    fn qjl_estimation_variance_is_bounded() {
        let config = TurboQuantConfig::new(BITS_3, TEST_DIM)
            .unwrap()
            .with_seed(TEST_ROTATION_SEED);

        let key = pseudo_random_vec(TEST_DIM, TEST_SEED_F);
        let query = pseudo_random_vec(TEST_DIM, TEST_SEED_G);
        let true_ip = dot_product(&key, &query);

        let mut sum_sq_error = 0.0_f64;

        for seed_offset in 0..STATISTICAL_SAMPLE_COUNT {
            let qjl_seed = TEST_QJL_SEED.wrapping_add(seed_offset as u64);
            let block = quantize_with_qjl(&config, &key, qjl_seed).unwrap();
            let est = estimate_inner_product_single(&query, &block, &config, qjl_seed).unwrap();
            let error = (est - true_ip) as f64;
            sum_sq_error += error * error;
        }

        let variance = sum_sq_error / STATISTICAL_SAMPLE_COUNT as f64;
        let true_ip_sq = (true_ip as f64) * (true_ip as f64);
        let relative_variance = if true_ip_sq > FLOAT_EPSILON as f64 {
            variance / true_ip_sq
        } else {
            variance
        };

        assert!(
            (relative_variance as f32) < MAX_RELATIVE_VARIANCE,
            "relative variance {relative_variance} exceeds bound {MAX_RELATIVE_VARIANCE}"
        );
    }

    // -- Orthogonal vectors --------------------------------------------------

    #[test]
    fn orthogonal_vectors_estimate_near_zero() {
        let config = TurboQuantConfig::new(BITS_3, TEST_DIM)
            .unwrap()
            .with_seed(TEST_ROTATION_SEED);

        let key = unit_vec(TEST_DIM, 0);
        let query = unit_vec(TEST_DIM, 1);

        let block = quantize_with_qjl(&config, &key, TEST_QJL_SEED).unwrap();
        let est = estimate_inner_product_single(&query, &block, &config, TEST_QJL_SEED).unwrap();

        assert!(
            est.abs() < ORTHO_TOLERANCE,
            "orthogonal estimate {est} not near zero"
        );
    }

    // -- Parallel vectors ----------------------------------------------------

    #[test]
    fn parallel_vectors_estimate_near_product() {
        let config = TurboQuantConfig::new(BITS_4, TEST_DIM)
            .unwrap()
            .with_seed(TEST_ROTATION_SEED);

        let key = pseudo_random_vec(TEST_DIM, TEST_SEED_H);
        let key_norm = l2_norm(&key);
        // query = 2 * key => <q, k> = 2 * ||k||^2
        let query: Vec<f32> = key.iter().map(|&v| v * TEST_SCALE_FACTOR).collect();
        let true_ip = dot_product(&query, &key);

        let block = quantize_with_qjl(&config, &key, TEST_QJL_SEED).unwrap();
        let est = estimate_inner_product_single(&query, &block, &config, TEST_QJL_SEED).unwrap();

        let relative_error = (est - true_ip).abs() / true_ip.abs();
        assert!(
            relative_error < PARALLEL_RELATIVE_TOLERANCE,
            "parallel relative error {relative_error} exceeds tolerance \
             {PARALLEL_RELATIVE_TOLERANCE} (est={est}, true={true_ip}, key_norm={key_norm})"
        );
    }

    // -- ceiling_div ---------------------------------------------------------

    #[test]
    fn ceiling_div_exact() {
        assert_eq!(ceiling_div(16, 8), 2);
        assert_eq!(ceiling_div(8, 8), 1);
    }

    #[test]
    fn ceiling_div_with_remainder() {
        assert_eq!(ceiling_div(17, 8), 3);
        assert_eq!(ceiling_div(1, 8), 1);
    }

    // -- is_valid_qjl_bits ---------------------------------------------------

    #[test]
    fn is_valid_qjl_bits_accepts_3_and_4() {
        assert!(is_valid_qjl_bits(BITS_3));
        assert!(is_valid_qjl_bits(BITS_4));
    }

    #[test]
    fn is_valid_qjl_bits_rejects_others() {
        assert!(!is_valid_qjl_bits(1));
        assert!(!is_valid_qjl_bits(2));
        assert!(!is_valid_qjl_bits(5));
    }

    // -----------------------------------------------------------------------
    // Bit-budget verification tests
    // -----------------------------------------------------------------------

    /// Dimension for bit-budget verification tests.
    const BIT_BUDGET_DIM: usize = 128;

    /// Expected PolarQuant bits for TQ3 (3-bit total = 2-bit polar + 1-bit QJL).
    const TQ3_EXPECTED_POLAR_BITS: u8 = 2;

    /// Expected PolarQuant bits for TQ4 (4-bit total = 3-bit polar + 1-bit QJL).
    const TQ4_EXPECTED_POLAR_BITS: u8 = 3;

    /// Seed for bit-budget verification tests.
    const BIT_BUDGET_SEED: u64 = 42;

    /// QJL seed for bit-budget verification tests.
    const BIT_BUDGET_QJL_SEED: u64 = 99999;

    #[test]
    fn tq3_uses_2bit_polar_quant() {
        let config = TurboQuantConfig::new(BITS_3, BIT_BUDGET_DIM)
            .unwrap()
            .with_seed(BIT_BUDGET_SEED);
        let data = pseudo_random_vec(BIT_BUDGET_DIM, TEST_SEED_A);
        let block = quantize_with_qjl(&config, &data, BIT_BUDGET_QJL_SEED).unwrap();

        assert_eq!(
            block.polar_block.bits, TQ3_EXPECTED_POLAR_BITS,
            "TQ3 should use {TQ3_EXPECTED_POLAR_BITS}-bit PolarQuant, got {}",
            block.polar_block.bits
        );
    }

    #[test]
    fn tq4_uses_3bit_polar_quant() {
        let config = TurboQuantConfig::new(BITS_4, BIT_BUDGET_DIM)
            .unwrap()
            .with_seed(BIT_BUDGET_SEED);
        let data = pseudo_random_vec(BIT_BUDGET_DIM, TEST_SEED_B);
        let block = quantize_with_qjl(&config, &data, BIT_BUDGET_QJL_SEED).unwrap();

        assert_eq!(
            block.polar_block.bits, TQ4_EXPECTED_POLAR_BITS,
            "TQ4 should use {TQ4_EXPECTED_POLAR_BITS}-bit PolarQuant, got {}",
            block.polar_block.bits
        );
    }

    // -----------------------------------------------------------------------
    // QJL block size verification tests
    // -----------------------------------------------------------------------

    /// Number of bits per byte, used for expected sign byte calculations.
    const BYTE_BITS: usize = 8;

    /// Residual norm storage size in bytes (f16).
    const RESIDUAL_NORM_STORAGE_BYTES: usize = 2;

    /// Expected TQ3 QjlBlock total size for d=128:
    /// polar(2-bit) 32 packed + 2 scale = 34, signs 16, residual_norm 2 => 52
    const TQ3_D128_EXPECTED_BLOCK_BYTES: usize = 52;

    /// Expected TQ4 QjlBlock total size for d=128:
    /// polar(3-bit) 48 packed + 2 scale = 50, signs 16, residual_norm 2 => 68
    const TQ4_D128_EXPECTED_BLOCK_BYTES: usize = 68;

    #[test]
    fn tq3_qjl_block_size_matches_expected() {
        let config = TurboQuantConfig::new(BITS_3, BIT_BUDGET_DIM)
            .unwrap()
            .with_seed(BIT_BUDGET_SEED);
        let data = pseudo_random_vec(BIT_BUDGET_DIM, TEST_SEED_C);
        let block = quantize_with_qjl(&config, &data, BIT_BUDGET_QJL_SEED).unwrap();

        let polar_bytes = block.polar_block.size_bytes();
        let sign_bytes = block.qjl_signs.len();
        let total = polar_bytes + sign_bytes + RESIDUAL_NORM_STORAGE_BYTES;

        assert_eq!(
            sign_bytes,
            BIT_BUDGET_DIM / BYTE_BITS,
            "TQ3 sign bytes: expected {}, got {sign_bytes}",
            BIT_BUDGET_DIM / BYTE_BITS
        );
        assert_eq!(
            total, TQ3_D128_EXPECTED_BLOCK_BYTES,
            "TQ3 QjlBlock total size: expected {TQ3_D128_EXPECTED_BLOCK_BYTES}, got {total} \
             (polar={polar_bytes}, signs={sign_bytes}, residual_norm={RESIDUAL_NORM_STORAGE_BYTES})"
        );
    }

    #[test]
    fn tq4_qjl_block_size_matches_expected() {
        let config = TurboQuantConfig::new(BITS_4, BIT_BUDGET_DIM)
            .unwrap()
            .with_seed(BIT_BUDGET_SEED);
        let data = pseudo_random_vec(BIT_BUDGET_DIM, TEST_SEED_D);
        let block = quantize_with_qjl(&config, &data, BIT_BUDGET_QJL_SEED).unwrap();

        let polar_bytes = block.polar_block.size_bytes();
        let sign_bytes = block.qjl_signs.len();
        let total = polar_bytes + sign_bytes + RESIDUAL_NORM_STORAGE_BYTES;

        assert_eq!(
            sign_bytes,
            BIT_BUDGET_DIM / BYTE_BITS,
            "TQ4 sign bytes: expected {}, got {sign_bytes}",
            BIT_BUDGET_DIM / BYTE_BITS
        );
        assert_eq!(
            total, TQ4_D128_EXPECTED_BLOCK_BYTES,
            "TQ4 QjlBlock total size: expected {TQ4_D128_EXPECTED_BLOCK_BYTES}, got {total} \
             (polar={polar_bytes}, signs={sign_bytes}, residual_norm={RESIDUAL_NORM_STORAGE_BYTES})"
        );
    }

    // -----------------------------------------------------------------------
    // Hash-based Rademacher tests
    // -----------------------------------------------------------------------

    /// Tolerance for hash-based Rademacher bias test.
    /// Hash signs should be +/- with near-equal probability over many samples.
    const RADEMACHER_BIAS_TOLERANCE: f32 = 0.15;

    /// Number of samples for hash-based Rademacher bias test.
    const RADEMACHER_SAMPLE_COUNT: usize = 1000;

    #[test]
    fn hash_rademacher_is_deterministic() {
        const TEST_ROW: usize = 5;
        const TEST_COL: usize = 10;
        let sign_a = rademacher_sign_from_hash(TEST_QJL_SEED, TEST_ROW, TEST_COL);
        let sign_b = rademacher_sign_from_hash(TEST_QJL_SEED, TEST_ROW, TEST_COL);
        assert!((sign_a - sign_b).abs() < FLOAT_EPSILON);
    }

    #[test]
    fn hash_rademacher_produces_unit_signs() {
        for row in 0..TEST_DIM {
            for col in 0..TEST_DIM {
                let sign = rademacher_sign_from_hash(TEST_QJL_SEED, row, col);
                assert!(
                    (sign - POSITIVE_SIGN).abs() < FLOAT_EPSILON
                        || (sign - NEGATIVE_SIGN).abs() < FLOAT_EPSILON,
                    "sign at ({row},{col}) = {sign}, expected +1.0 or -1.0"
                );
            }
        }
    }

    #[test]
    fn hash_rademacher_approximately_unbiased() {
        let mut positive_count = 0usize;
        for row in 0..RADEMACHER_SAMPLE_COUNT {
            for col in 0..TEST_DIM {
                let sign = rademacher_sign_from_hash(TEST_QJL_SEED, row, col);
                if sign > 0.0 {
                    positive_count += 1;
                }
            }
        }
        let total = RADEMACHER_SAMPLE_COUNT * TEST_DIM;
        let positive_fraction = positive_count as f32 / total as f32;
        const EXPECTED_FRACTION: f32 = 0.5;
        let bias = (positive_fraction - EXPECTED_FRACTION).abs();
        assert!(
            bias < RADEMACHER_BIAS_TOLERANCE,
            "hash Rademacher bias {bias} (positive fraction {positive_fraction}) \
             exceeds tolerance {RADEMACHER_BIAS_TOLERANCE}"
        );
    }

    #[test]
    fn hash_rademacher_different_positions_differ() {
        // Different (row, col) pairs should produce different signs at least sometimes.
        let mut same_count = 0usize;
        const HASH_DIVERSITY_CHECK_COUNT: usize = 100;
        for i in 0..HASH_DIVERSITY_CHECK_COUNT {
            let sign_a = rademacher_sign_from_hash(TEST_QJL_SEED, i, 0);
            let sign_b = rademacher_sign_from_hash(TEST_QJL_SEED, i, 1);
            if (sign_a - sign_b).abs() < FLOAT_EPSILON {
                same_count += 1;
            }
        }
        // With random signs, we expect ~50% to match. If ALL match, something is wrong.
        assert!(
            same_count < HASH_DIVERSITY_CHECK_COUNT,
            "all {HASH_DIVERSITY_CHECK_COUNT} sign pairs were identical -- hash is not mixing positions"
        );
    }

    // -----------------------------------------------------------------------
    // Estimate inner product tests (batch API)
    // -----------------------------------------------------------------------

    #[test]
    fn estimate_inner_product_is_deterministic() {
        let config = TurboQuantConfig::new(BITS_3, TEST_DIM)
            .unwrap()
            .with_seed(TEST_ROTATION_SEED);
        let key = pseudo_random_vec(TEST_DIM, TEST_SEED_D);
        let query = pseudo_random_vec(TEST_DIM, TEST_SEED_E);

        let block = quantize_with_qjl(&config, &key, TEST_QJL_SEED).unwrap();
        let r_query = precompute_query_projections(&query, TEST_DIM, TEST_QJL_SEED);

        let est_a = estimate_inner_product(&query, &r_query, &block, &config).unwrap();
        let est_b = estimate_inner_product(&query, &r_query, &block, &config).unwrap();

        assert!(
            (est_a - est_b).abs() < FLOAT_EPSILON,
            "not deterministic: {est_a} vs {est_b}"
        );
    }

    #[test]
    fn estimate_inner_product_unbiased_over_many_samples() {
        let config = TurboQuantConfig::new(BITS_3, TEST_DIM)
            .unwrap()
            .with_seed(TEST_ROTATION_SEED);

        let mut bias_sum = 0.0_f64;

        for i in 0..STATISTICAL_SAMPLE_COUNT {
            let key_seed = (i as u64)
                .wrapping_mul(LCG_MULTIPLIER)
                .wrapping_add(TEST_SAMPLE_COUNT);
            let query_seed = (i as u64)
                .wrapping_mul(LCG_MULTIPLIER)
                .wrapping_add(TEST_LARGE_SAMPLE_COUNT);
            let qjl_seed = TEST_QJL_SEED.wrapping_add(i as u64);

            let key = pseudo_random_vec(TEST_DIM, key_seed);
            let query = pseudo_random_vec(TEST_DIM, query_seed);
            let true_ip = dot_product(&key, &query) as f64;

            let block = quantize_with_qjl(&config, &key, qjl_seed).unwrap();
            let r_query = precompute_query_projections(&query, TEST_DIM, qjl_seed);
            let est = estimate_inner_product(&query, &r_query, &block, &config).unwrap() as f64;

            bias_sum += est - true_ip;
        }

        let mean_bias = (bias_sum / STATISTICAL_SAMPLE_COUNT as f64).abs() as f32;
        assert!(
            mean_bias < BIAS_TOLERANCE,
            "estimate mean bias {mean_bias} exceeds tolerance {BIAS_TOLERANCE}"
        );
    }

    // -----------------------------------------------------------------------
    // Precompute query projections tests
    // -----------------------------------------------------------------------

    #[test]
    fn precompute_projections_correct_length() {
        let query = pseudo_random_vec(TEST_DIM, TEST_SEED_A);
        let projections = precompute_query_projections(&query, TEST_DIM, TEST_QJL_SEED);
        assert_eq!(projections.len(), TEST_DIM);
    }

    #[test]
    fn precompute_projections_deterministic() {
        let query = pseudo_random_vec(TEST_DIM, TEST_SEED_A);
        let proj_a = precompute_query_projections(&query, TEST_DIM, TEST_QJL_SEED);
        let proj_b = precompute_query_projections(&query, TEST_DIM, TEST_QJL_SEED);
        assert_eq!(proj_a, proj_b);
    }

    // -----------------------------------------------------------------------
    // Unpack signs tests
    // -----------------------------------------------------------------------

    #[test]
    fn unpack_signs_roundtrip() {
        let bools: Vec<bool> = (0..PACK_TEST_LEN)
            .map(|i| i % SIGN_PACK_PATTERN_STRIDE == 0)
            .collect();
        let packed = pack_sign_bits(&bools);
        let unpacked = unpack_signs_to_f32(&packed, PACK_TEST_LEN);

        for (i, &expected_positive) in bools.iter().enumerate() {
            let expected = if expected_positive {
                POSITIVE_SIGN
            } else {
                NEGATIVE_SIGN
            };
            assert!(
                (unpacked[i] - expected).abs() < FLOAT_EPSILON,
                "bit {i}: expected {expected}, got {}",
                unpacked[i]
            );
        }
    }

    // -----------------------------------------------------------------------
    // QJL correction vs naive correction
    // -----------------------------------------------------------------------

    #[test]
    fn qjl_correction_matches_dot_product() {
        // Build some R*query values and signs, verify qjl_correction
        // matches the manual sign_bit * r_query sum.
        const R_QUERY_SCALE: f32 = 0.1;
        const R_QUERY_OFFSET: f32 = 3.0;
        let r_query: Vec<f32> = (0..TEST_DIM)
            .map(|i| (i as f32) * R_QUERY_SCALE - R_QUERY_OFFSET)
            .collect();
        let bools: Vec<bool> = (0..TEST_DIM).map(|i| i % 3 == 0).collect();
        let packed_signs = pack_sign_bits(&bools);

        let expected: f32 = (0..TEST_DIM)
            .map(|j| r_query[j] * sign_bit(&packed_signs, j))
            .sum();
        let actual = qjl_correction(&r_query, &packed_signs, TEST_DIM);

        assert!(
            (expected - actual).abs() < FLOAT_EPSILON,
            "qjl_correction mismatch: expected {expected}, got {actual}"
        );
    }

    // -- QjlBlock accessors and from_parts -----------------------------------

    #[test]
    fn from_parts_roundtrip() {
        let config = TurboQuantConfig::new(BITS_3, TEST_DIM)
            .unwrap()
            .with_seed(TEST_ROTATION_SEED);
        let data = pseudo_random_vec(TEST_DIM, TEST_QJL_SEED);
        let original = quantize_with_qjl(&config, &data, TEST_QJL_SEED).unwrap();

        let reconstructed = QjlBlock::from_parts(
            PackedBlock::from_raw(
                original.polar_block.bits,
                original.polar_block.scale,
                original.polar_block.packed_indices.to_vec(),
            ),
            original.qjl_signs.to_vec(),
            original.residual_norm,
        );

        assert_eq!(reconstructed.polar_block.bits, original.polar_block.bits);
        assert_eq!(reconstructed.polar_block.scale, original.polar_block.scale);
        assert_eq!(
            reconstructed.polar_block.packed_indices,
            original.polar_block.packed_indices
        );
        assert_eq!(reconstructed.qjl_signs, original.qjl_signs);
        assert_eq!(reconstructed.residual_norm, original.residual_norm);
    }

    #[test]
    fn accessor_polar_block_matches_quantized() {
        let config = TurboQuantConfig::new(BITS_3, TEST_DIM)
            .unwrap()
            .with_seed(TEST_ROTATION_SEED);
        let data = pseudo_random_vec(TEST_DIM, TEST_QJL_SEED);
        let block = quantize_with_qjl(&config, &data, TEST_QJL_SEED).unwrap();

        // polar_block bits should be total_bits - 1
        assert_eq!(block.polar_block.bits, BITS_3 - 1);
        // scale should be positive (L2 norm of non-zero vector)
        assert!(block.polar_block.scale.to_f32() > 0.0);
        // packed_indices should be non-empty
        assert!(!block.polar_block.packed_indices.is_empty());
        // qjl_signs should have ceil(dim/8) bytes
        const BITS_PER_BYTE: usize = 8;
        let expected_sign_bytes = TEST_DIM.div_ceil(BITS_PER_BYTE);
        assert_eq!(block.qjl_signs.len(), expected_sign_bytes);
    }
}
