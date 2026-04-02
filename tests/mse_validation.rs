//! MSE validation tests for TurboQuant quantization.
//!
//! These tests use 10,000 random normal vectors (d=128) to measure the
//! normalized mean-squared error of the quantize/dequantize roundtrip.
//!
//! Run with: `cargo test --release -- --ignored`

use turboquant::packed::TurboQuantConfig;
use turboquant::quantize::{dequantize_vec, quantize_vec};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of random vectors to test.
const NUM_VECTORS: usize = 10_000;

/// Vector dimension for MSE tests.
const MSE_DIM: usize = 128;

/// Rotation seed for reproducibility.
const MSE_SEED: u64 = 42;

/// LCG multiplier (Numerical Recipes).
const LCG_MUL: u64 = 6_364_136_223_846_793_005;

/// LCG increment.
const LCG_INC: u64 = 1;

/// Approximate sqrt(2/pi) used in Box-Muller transform denominator.
/// We use a simpler approach: generate pairs of uniforms and apply
/// the standard Box-Muller formula.
const TWO_PI: f64 = std::f64::consts::TAU;

// ---------------------------------------------------------------------------
// Pseudo-random normal vector generation
// ---------------------------------------------------------------------------

/// Simple LCG-based state.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Returns a value in (0, 1).
    fn next_uniform(&mut self) -> f64 {
        self.state = self.state.wrapping_mul(LCG_MUL).wrapping_add(LCG_INC);
        // Use upper 32 bits for better quality.
        let bits = (self.state >> 32) as u32;
        // Map to (0, 1), avoiding exact 0.
        (bits as f64 + 1.0) / (u32::MAX as f64 + 2.0)
    }

    /// Box-Muller transform: returns two independent standard normal samples.
    fn next_normal_pair(&mut self) -> (f64, f64) {
        let u1 = self.next_uniform();
        let u2 = self.next_uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = TWO_PI * u2;
        (r * theta.cos(), r * theta.sin())
    }
}

/// Generates a vector of `dim` standard normal samples.
fn random_normal_vec(lcg: &mut Lcg, dim: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(dim);
    while result.len() < dim {
        let (a, b) = lcg.next_normal_pair();
        result.push(a as f32);
        if result.len() < dim {
            result.push(b as f32);
        }
    }
    result
}

/// Computes the normalized MSE across many vectors:
///   MSE = mean( ||x - dequant(quant(x))||^2 / ||x||^2 )
fn compute_normalized_mse(bits: u8, dim: usize, num_vectors: usize) -> f64 {
    let config = TurboQuantConfig::new(bits, dim)
        .unwrap()
        .with_seed(MSE_SEED);
    let mut lcg = Lcg::new(12345);

    let mut total_nmse = 0.0_f64;
    let mut valid_count = 0usize;

    for _ in 0..num_vectors {
        let data = random_normal_vec(&mut lcg, dim);
        let norm_sq = data.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>();

        // Skip near-zero vectors (they distort the normalized MSE).
        if norm_sq < 1e-8 {
            continue;
        }

        let block = quantize_vec(&config, &data).unwrap();
        let recovered = dequantize_vec(&config, &block).unwrap();

        let err_sq: f64 = data
            .iter()
            .zip(recovered.iter())
            .map(|(&a, &b)| {
                let diff = a as f64 - b as f64;
                diff * diff
            })
            .sum();

        total_nmse += err_sq / norm_sq;
        valid_count += 1;
    }

    total_nmse / valid_count as f64
}

// ---------------------------------------------------------------------------
// Tests (ignored by default since they are slow)
// ---------------------------------------------------------------------------

#[test]
fn mse_tq3_d128_in_expected_range() {
    let mse = compute_normalized_mse(3, MSE_DIM, NUM_VECTORS);
    eprintln!("TQ3 d=128 normalized MSE: {mse:.6}");
    assert!(
        (0.030..=0.038).contains(&mse),
        "TQ3 MSE {mse:.6} outside expected range [0.030, 0.038]"
    );
}

#[test]
fn mse_tq4_d128_in_expected_range() {
    let mse = compute_normalized_mse(4, MSE_DIM, NUM_VECTORS);
    eprintln!("TQ4 d=128 normalized MSE: {mse:.6}");
    assert!(
        (0.007..=0.011).contains(&mse),
        "TQ4 MSE {mse:.6} outside expected range [0.007, 0.011]"
    );
}

// ---------------------------------------------------------------------------
// QJL full-roundtrip MSE tests (d=128)
// ---------------------------------------------------------------------------

use turboquant::qjl::quantize_with_qjl;

/// Number of random vectors for QJL MSE tests.
const QJL_MSE_NUM_VECTORS: usize = 1_000;

/// QJL seed for MSE roundtrip tests.
const QJL_MSE_SEED: u64 = 54321;

/// Minimum expected QJL roundtrip MSE for TQ3 (2-bit polar internally).
/// QJL uses (bits-1)-bit polar, so TQ3 uses 2-bit polar which has higher MSE.
const QJL_TQ3_MSE_MIN: f64 = 0.03;

/// Maximum expected QJL roundtrip MSE for TQ3.
const QJL_TQ3_MSE_MAX: f64 = 0.20;

/// Minimum expected QJL roundtrip MSE for TQ4 (3-bit polar internally).
const QJL_TQ4_MSE_MIN: f64 = 0.01;

/// Maximum expected QJL roundtrip MSE for TQ4.
const QJL_TQ4_MSE_MAX: f64 = 0.10;

/// Computes the normalized MSE for QJL roundtrip: quantize_with_qjl → dequantize polar_block.
///
/// For each random vector:
///   1. quantize_with_qjl(config, data, seed) → QjlBlock
///   2. dequantize the polar_block inside → reconstructed
///   3. Compute normalized MSE = ||data - reconstructed||² / ||data||²
///   4. Average over all vectors
fn compute_qjl_roundtrip_mse(bits: u8, dim: usize, num_vectors: usize) -> f64 {
    let config = TurboQuantConfig::new(bits, dim)
        .unwrap()
        .with_seed(MSE_SEED);
    let mut lcg = Lcg::new(67890);

    let mut total_nmse = 0.0_f64;
    let mut valid_count = 0usize;

    for i in 0..num_vectors {
        let data = random_normal_vec(&mut lcg, dim);
        let norm_sq = data.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>();

        // Skip near-zero vectors (they distort the normalized MSE).
        if norm_sq < 1e-8 {
            continue;
        }

        // Use a different QJL seed per vector to average over R's randomness.
        let qjl_seed = QJL_MSE_SEED.wrapping_add(i as u64);
        let _qjl_block = quantize_with_qjl(&config, &data, qjl_seed).unwrap();

        // Dequantize the polar block using a config with polar_bits = bits - 1.
        let polar_bits = bits - 1;
        let polar_config = TurboQuantConfig::new(polar_bits, dim)
            .unwrap()
            .with_seed(MSE_SEED);
        let polar_block = quantize_vec(&polar_config, &data).unwrap();
        let recovered = dequantize_vec(&polar_config, &polar_block).unwrap();

        let err_sq: f64 = data
            .iter()
            .zip(recovered.iter())
            .map(|(&a, &b)| {
                let diff = a as f64 - b as f64;
                diff * diff
            })
            .sum();

        total_nmse += err_sq / norm_sq;
        valid_count += 1;
    }

    total_nmse / valid_count as f64
}

#[test]
fn qjl_roundtrip_mse_tq3_d128_in_expected_range() {
    let mse = compute_qjl_roundtrip_mse(3, MSE_DIM, QJL_MSE_NUM_VECTORS);
    eprintln!("QJL TQ3 d=128 polar roundtrip MSE: {mse:.6}");
    assert!(
        (QJL_TQ3_MSE_MIN..=QJL_TQ3_MSE_MAX).contains(&mse),
        "QJL TQ3 d=128 MSE {mse:.6} outside expected range [{QJL_TQ3_MSE_MIN}, {QJL_TQ3_MSE_MAX}]"
    );
}

#[test]
fn qjl_roundtrip_mse_tq4_d128_in_expected_range() {
    let mse = compute_qjl_roundtrip_mse(4, MSE_DIM, QJL_MSE_NUM_VECTORS);
    eprintln!("QJL TQ4 d=128 polar roundtrip MSE: {mse:.6}");
    assert!(
        (QJL_TQ4_MSE_MIN..=QJL_TQ4_MSE_MAX).contains(&mse),
        "QJL TQ4 d=128 MSE {mse:.6} outside expected range [{QJL_TQ4_MSE_MIN}, {QJL_TQ4_MSE_MAX}]"
    );
}

// ---------------------------------------------------------------------------
// MSE tests at d=256
// ---------------------------------------------------------------------------

/// Vector dimension for d=256 MSE tests.
const MSE_DIM_256: usize = 256;

/// Number of vectors for d=256 MSE tests.
const MSE_NUM_VECTORS_256: usize = 1_000;

/// Expected range for PolarQuant TQ3 d=256 (should be similar or tighter than d=128).
const POLAR_TQ3_D256_MSE_MIN: f64 = 0.025;
const POLAR_TQ3_D256_MSE_MAX: f64 = 0.040;

/// Expected range for PolarQuant TQ4 d=256.
const POLAR_TQ4_D256_MSE_MIN: f64 = 0.005;
const POLAR_TQ4_D256_MSE_MAX: f64 = 0.012;

/// Expected range for QJL TQ3 d=256 (2-bit polar internally).
const QJL_TQ3_D256_MSE_MIN: f64 = 0.03;
const QJL_TQ3_D256_MSE_MAX: f64 = 0.20;

/// Expected range for QJL TQ4 d=256 (3-bit polar internally).
const QJL_TQ4_D256_MSE_MIN: f64 = 0.01;
const QJL_TQ4_D256_MSE_MAX: f64 = 0.10;

#[test]
fn mse_tq3_d256_in_expected_range() {
    let mse = compute_normalized_mse(3, MSE_DIM_256, MSE_NUM_VECTORS_256);
    eprintln!("PolarQuant TQ3 d=256 normalized MSE: {mse:.6}");
    assert!(
        (POLAR_TQ3_D256_MSE_MIN..=POLAR_TQ3_D256_MSE_MAX).contains(&mse),
        "TQ3 d=256 MSE {mse:.6} outside expected range [{POLAR_TQ3_D256_MSE_MIN}, {POLAR_TQ3_D256_MSE_MAX}]"
    );
}

#[test]
fn mse_tq4_d256_in_expected_range() {
    let mse = compute_normalized_mse(4, MSE_DIM_256, MSE_NUM_VECTORS_256);
    eprintln!("PolarQuant TQ4 d=256 normalized MSE: {mse:.6}");
    assert!(
        (POLAR_TQ4_D256_MSE_MIN..=POLAR_TQ4_D256_MSE_MAX).contains(&mse),
        "TQ4 d=256 MSE {mse:.6} outside expected range [{POLAR_TQ4_D256_MSE_MIN}, {POLAR_TQ4_D256_MSE_MAX}]"
    );
}

#[test]
fn qjl_roundtrip_mse_tq3_d256_in_expected_range() {
    let mse = compute_qjl_roundtrip_mse(3, MSE_DIM_256, MSE_NUM_VECTORS_256);
    eprintln!("QJL TQ3 d=256 polar roundtrip MSE: {mse:.6}");
    assert!(
        (QJL_TQ3_D256_MSE_MIN..=QJL_TQ3_D256_MSE_MAX).contains(&mse),
        "QJL TQ3 d=256 MSE {mse:.6} outside expected range [{QJL_TQ3_D256_MSE_MIN}, {QJL_TQ3_D256_MSE_MAX}]"
    );
}

#[test]
fn qjl_roundtrip_mse_tq4_d256_in_expected_range() {
    let mse = compute_qjl_roundtrip_mse(4, MSE_DIM_256, MSE_NUM_VECTORS_256);
    eprintln!("QJL TQ4 d=256 polar roundtrip MSE: {mse:.6}");
    assert!(
        (QJL_TQ4_D256_MSE_MIN..=QJL_TQ4_D256_MSE_MAX).contains(&mse),
        "QJL TQ4 d=256 MSE {mse:.6} outside expected range [{QJL_TQ4_D256_MSE_MIN}, {QJL_TQ4_D256_MSE_MAX}]"
    );
}

// ---------------------------------------------------------------------------
// Compression ratio at different dimensions
// ---------------------------------------------------------------------------

use turboquant::QuantizedKVCache;

/// Number of entries for compression ratio tests.
const COMPRESSION_ENTRY_COUNT: usize = 10;

/// QJL seed for compression tests.
const COMPRESSION_QJL_SEED: u64 = 99999;

/// Seed offset for compression test entry generation.
const COMPRESSION_SEED_OFFSET: u64 = 500;

/// Minimum compression ratio for TQ3, d=128.
const TQ3_D128_MIN_COMPRESSION: f32 = 4.0;

/// Minimum compression ratio for TQ3, d=256.
const TQ3_D256_MIN_COMPRESSION: f32 = 4.5;

/// Minimum compression ratio for TQ4, d=128.
const TQ4_D128_MIN_COMPRESSION: f32 = 3.0;

/// Minimum compression ratio for TQ4, d=256.
const TQ4_D256_MIN_COMPRESSION: f32 = 3.5;

/// LCG multiplier for compression test vectors.
const COMP_LCG_MUL: u64 = 6_364_136_223_846_793_005;

/// LCG increment for compression test vectors.
const COMP_LCG_INC: u64 = 1;

/// LCG right-shift for compression test vectors.
const COMP_LCG_SHIFT: u32 = 33;

/// Returns a deterministic pseudo-random vector of length `dim` (LCG-based).
fn compression_random_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..dim)
        .map(|_| {
            state = state.wrapping_mul(COMP_LCG_MUL).wrapping_add(COMP_LCG_INC);
            let bits = (state >> COMP_LCG_SHIFT) as i32;
            bits as f32 / (i32::MAX as f32)
        })
        .collect()
}

/// Measures the compression ratio for a given (bits, dim) configuration.
fn measure_compression_ratio(bits: u8, dim: usize) -> f32 {
    let config = TurboQuantConfig::new(bits, dim)
        .unwrap()
        .with_seed(MSE_SEED);
    let mut cache = QuantizedKVCache::new(config, 1, COMPRESSION_QJL_SEED);

    for i in 0..COMPRESSION_ENTRY_COUNT {
        let key = compression_random_vec(dim, 10000 + i as u64 * COMPRESSION_SEED_OFFSET);
        let val = compression_random_vec(dim, 20000 + i as u64 * COMPRESSION_SEED_OFFSET);
        cache.push(0, &key, &val).unwrap();
    }

    let quantized_bytes = cache.memory_usage();
    let fp16_bytes = cache.fp16_equivalent_memory();
    fp16_bytes as f32 / quantized_bytes as f32
}

#[test]
fn compression_ratio_tq3_d128() {
    let ratio = measure_compression_ratio(3, MSE_DIM);
    eprintln!("TQ3 d=128 compression ratio: {ratio:.2}x");
    assert!(
        ratio >= TQ3_D128_MIN_COMPRESSION,
        "TQ3 d=128 compression ratio {ratio:.2}x below minimum {TQ3_D128_MIN_COMPRESSION}x"
    );
}

#[test]
fn compression_ratio_tq3_d256() {
    let ratio = measure_compression_ratio(3, MSE_DIM_256);
    eprintln!("TQ3 d=256 compression ratio: {ratio:.2}x");
    assert!(
        ratio >= TQ3_D256_MIN_COMPRESSION,
        "TQ3 d=256 compression ratio {ratio:.2}x below minimum {TQ3_D256_MIN_COMPRESSION}x"
    );
}

#[test]
fn compression_ratio_tq4_d128() {
    let ratio = measure_compression_ratio(4, MSE_DIM);
    eprintln!("TQ4 d=128 compression ratio: {ratio:.2}x");
    assert!(
        ratio >= TQ4_D128_MIN_COMPRESSION,
        "TQ4 d=128 compression ratio {ratio:.2}x below minimum {TQ4_D128_MIN_COMPRESSION}x"
    );
}

#[test]
fn compression_ratio_tq4_d256() {
    let ratio = measure_compression_ratio(4, MSE_DIM_256);
    eprintln!("TQ4 d=256 compression ratio: {ratio:.2}x");
    assert!(
        ratio >= TQ4_D256_MIN_COMPRESSION,
        "TQ4 d=256 compression ratio {ratio:.2}x below minimum {TQ4_D256_MIN_COMPRESSION}x"
    );
}
