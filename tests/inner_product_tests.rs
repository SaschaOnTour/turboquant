//! Integration tests for QJL inner-product estimation.
//!
//! Tests the end-to-end unbiasedness and variance properties of the
//! TURBOQUANTprod algorithm (Algorithm 2) across many random vector pairs.

use turboquant::packed::TurboQuantConfig;
use turboquant::qjl::{dot_product, estimate_inner_product_single, quantize_with_qjl};

// ---------------------------------------------------------------------------
// Named constants (no magic numbers)
// ---------------------------------------------------------------------------

/// Dimension for integration tests (power of two for WHT).
const TEST_DIM: usize = 64;

/// Rotation seed for PolarQuant.
const ROTATION_SEED: u64 = 42;

/// QJL Rademacher matrix seed.
const QJL_SEED: u64 = 12345;

/// Number of random pairs for the large statistical test.
const LARGE_SAMPLE_COUNT: usize = 10_000;

/// Number of random pairs for the quick statistical test.
const QUICK_SAMPLE_COUNT: usize = 200;

/// Maximum acceptable mean bias (absolute) for the bias test.
const BIAS_TOLERANCE: f32 = 0.02;

/// Maximum acceptable relative variance.
const MAX_RELATIVE_VARIANCE: f64 = 2.0;

/// LCG multiplier for pseudo-random vector generation.
const LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;

/// LCG increment for pseudo-random vector generation.
const LCG_INCREMENT: u64 = 1;

/// Right-shift for extracting bits from LCG state.
const LCG_SHIFT: u32 = 33;

/// Overall bit budget (3-bit: 2-bit polar + 1-bit QJL).
const BITS_3: u8 = 3;

/// Key seed offset to separate key and query generation.
const KEY_SEED_OFFSET: u64 = 1000;

/// Query seed offset to separate key and query generation.
const QUERY_SEED_OFFSET: u64 = 2000;

// ---------------------------------------------------------------------------
// Helper: deterministic pseudo-random vector
// ---------------------------------------------------------------------------

/// Returns a deterministic pseudo-random vector of length `dim`.
fn pseudo_random_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..dim)
        .map(|_| {
            state = state
                .wrapping_mul(LCG_MULTIPLIER)
                .wrapping_add(LCG_INCREMENT);
            let bits = (state >> LCG_SHIFT) as i32;
            bits as f32 / (i32::MAX as f32)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Integration tests
// ---------------------------------------------------------------------------

/// Large-scale bias test: 10,000 random (key, query) pairs.
///
/// Verifies that the QJL-corrected inner product estimate is unbiased
/// in expectation: |mean(estimate - true)| < tolerance.
#[test]
fn qjl_inner_product_bias_10k_pairs() {
    let config = TurboQuantConfig::new(BITS_3, TEST_DIM)
        .unwrap()
        .with_seed(ROTATION_SEED);

    let mut bias_sum = 0.0_f64;

    for i in 0..LARGE_SAMPLE_COUNT {
        let key_seed = (i as u64)
            .wrapping_mul(LCG_MULTIPLIER)
            .wrapping_add(KEY_SEED_OFFSET);
        let query_seed = (i as u64)
            .wrapping_mul(LCG_MULTIPLIER)
            .wrapping_add(QUERY_SEED_OFFSET);
        // Different QJL seed per sample to average over R's randomness.
        let qjl_seed = QJL_SEED.wrapping_add(i as u64);

        let key = pseudo_random_vec(TEST_DIM, key_seed);
        let query = pseudo_random_vec(TEST_DIM, query_seed);
        let true_ip = dot_product(&key, &query) as f64;

        let block = quantize_with_qjl(&config, &key, qjl_seed).unwrap();
        let est = estimate_inner_product_single(&query, &block, &config, qjl_seed).unwrap() as f64;

        bias_sum += est - true_ip;
    }

    let mean_bias = (bias_sum / LARGE_SAMPLE_COUNT as f64).abs() as f32;
    assert!(
        mean_bias < BIAS_TOLERANCE,
        "mean bias {mean_bias} exceeds tolerance {BIAS_TOLERANCE} over {LARGE_SAMPLE_COUNT} pairs"
    );
}

/// Quick bias test (not ignored): smaller sample to run in CI.
#[test]
fn qjl_inner_product_bias_quick() {
    let config = TurboQuantConfig::new(BITS_3, TEST_DIM)
        .unwrap()
        .with_seed(ROTATION_SEED);

    let mut bias_sum = 0.0_f64;
    let looser_tolerance: f32 = 0.1; // looser for smaller sample

    for i in 0..QUICK_SAMPLE_COUNT {
        let key_seed = (i as u64)
            .wrapping_mul(LCG_MULTIPLIER)
            .wrapping_add(KEY_SEED_OFFSET);
        let query_seed = (i as u64)
            .wrapping_mul(LCG_MULTIPLIER)
            .wrapping_add(QUERY_SEED_OFFSET);
        // Different QJL seed per sample to average over R's randomness.
        let qjl_seed = QJL_SEED.wrapping_add(i as u64);

        let key = pseudo_random_vec(TEST_DIM, key_seed);
        let query = pseudo_random_vec(TEST_DIM, query_seed);
        let true_ip = dot_product(&key, &query) as f64;

        let block = quantize_with_qjl(&config, &key, qjl_seed).unwrap();
        let est = estimate_inner_product_single(&query, &block, &config, qjl_seed).unwrap() as f64;

        bias_sum += est - true_ip;
    }

    let mean_bias = (bias_sum / QUICK_SAMPLE_COUNT as f64).abs() as f32;
    assert!(
        mean_bias < looser_tolerance,
        "mean bias {mean_bias} exceeds tolerance {looser_tolerance} over {QUICK_SAMPLE_COUNT} pairs"
    );
}

/// Variance check: the estimation error should have bounded variance.
#[test]
fn qjl_inner_product_variance_bounded() {
    let config = TurboQuantConfig::new(BITS_3, TEST_DIM)
        .unwrap()
        .with_seed(ROTATION_SEED);

    let mut sum_sq_error = 0.0_f64;
    let mut sum_true_sq = 0.0_f64;

    for i in 0..QUICK_SAMPLE_COUNT {
        let key_seed = (i as u64)
            .wrapping_mul(LCG_MULTIPLIER)
            .wrapping_add(KEY_SEED_OFFSET);
        let query_seed = (i as u64)
            .wrapping_mul(LCG_MULTIPLIER)
            .wrapping_add(QUERY_SEED_OFFSET);
        // Different QJL seed per sample to average over R's randomness.
        let qjl_seed = QJL_SEED.wrapping_add(i as u64);

        let key = pseudo_random_vec(TEST_DIM, key_seed);
        let query = pseudo_random_vec(TEST_DIM, query_seed);
        let true_ip = dot_product(&key, &query) as f64;

        let block = quantize_with_qjl(&config, &key, qjl_seed).unwrap();
        let est = estimate_inner_product_single(&query, &block, &config, qjl_seed).unwrap() as f64;

        let error = est - true_ip;
        sum_sq_error += error * error;
        sum_true_sq += true_ip * true_ip;
    }

    let mean_sq_error = sum_sq_error / QUICK_SAMPLE_COUNT as f64;
    let mean_true_sq = sum_true_sq / QUICK_SAMPLE_COUNT as f64;

    let relative_variance = if mean_true_sq > 1e-10 {
        mean_sq_error / mean_true_sq
    } else {
        mean_sq_error
    };

    assert!(
        relative_variance < MAX_RELATIVE_VARIANCE,
        "relative variance {relative_variance} exceeds bound {MAX_RELATIVE_VARIANCE}"
    );
}

// ---------------------------------------------------------------------------
// End-to-end accuracy test via QuantizedKVCache API
// ---------------------------------------------------------------------------

use turboquant::QuantizedKVCache;

/// Number of KV pairs pushed in the end-to-end cache accuracy test.
const E2E_ENTRY_COUNT: usize = 100;

/// Dimension for end-to-end test (power of two, realistic head size).
const E2E_DIM: usize = 128;

/// Seed offset for end-to-end test key generation.
const E2E_KEY_SEED_OFFSET: u64 = 5000;

/// Seed offset for end-to-end test value generation.
const E2E_VALUE_SEED_OFFSET: u64 = 6000;

/// Seed offset for end-to-end test query generation.
const E2E_QUERY_SEED_OFFSET: u64 = 8000;

/// Number of layers in end-to-end cache test.
const E2E_NUM_LAYERS: usize = 1;

/// Layer index for end-to-end test.
const E2E_LAYER: usize = 0;

/// Maximum acceptable mean relative error for attention scores.
/// Over 100 entries the mean should converge to a reasonable value.
const E2E_MAX_MEAN_RELATIVE_ERROR: f64 = 0.5;

/// Maximum acceptable absolute mean error (bias) over all entries.
/// Should be close to zero for an unbiased estimator.
const E2E_BIAS_TOLERANCE: f64 = 0.05;

/// QJL seed for end-to-end test.
const E2E_QJL_SEED: u64 = 54321;

/// Rotation seed for end-to-end test.
const E2E_ROTATION_SEED: u64 = 42;

/// End-to-end test: push 100 KV pairs through QuantizedKVCache, then verify
/// that attention scores are unbiased and have bounded error relative to the
/// true dot products.
#[test]
fn e2e_kv_cache_attention_scores_unbiased() {
    let config = TurboQuantConfig::new(BITS_3, E2E_DIM)
        .unwrap()
        .with_seed(E2E_ROTATION_SEED);
    let mut cache = QuantizedKVCache::new(config, E2E_NUM_LAYERS, E2E_QJL_SEED);

    // Store original keys for ground-truth comparison.
    let mut original_keys: Vec<Vec<f32>> = Vec::with_capacity(E2E_ENTRY_COUNT);

    for i in 0..E2E_ENTRY_COUNT {
        let key_seed = (i as u64)
            .wrapping_mul(LCG_MULTIPLIER)
            .wrapping_add(E2E_KEY_SEED_OFFSET);
        let val_seed = (i as u64)
            .wrapping_mul(LCG_MULTIPLIER)
            .wrapping_add(E2E_VALUE_SEED_OFFSET);

        let key = pseudo_random_vec(E2E_DIM, key_seed);
        let val = pseudo_random_vec(E2E_DIM, val_seed);

        original_keys.push(key.clone());
        cache.push(E2E_LAYER, &key, &val).unwrap();
    }

    // Generate a query and compute attention scores.
    let query = pseudo_random_vec(E2E_DIM, E2E_QUERY_SEED_OFFSET);
    let scores = cache.attention_scores(E2E_LAYER, &query).unwrap();

    assert_eq!(
        scores.len(),
        E2E_ENTRY_COUNT,
        "Expected {} scores, got {}",
        E2E_ENTRY_COUNT,
        scores.len()
    );

    // Compute true dot products and compare.
    let mut error_sum = 0.0_f64;
    let mut abs_error_sum = 0.0_f64;
    let mut true_sq_sum = 0.0_f64;

    let compute_errors = |error_sum: &mut f64, abs_error_sum: &mut f64, true_sq_sum: &mut f64| {
        for (i, &score) in scores.iter().enumerate() {
            let true_ip = dot_product(&query, &original_keys[i]) as f64;
            let error = score as f64 - true_ip;
            *error_sum += error;
            *abs_error_sum += error.abs();
            *true_sq_sum += true_ip * true_ip;
        }
    };
    compute_errors(&mut error_sum, &mut abs_error_sum, &mut true_sq_sum);

    let mean_error = error_sum / E2E_ENTRY_COUNT as f64;
    let mean_abs_error = abs_error_sum / E2E_ENTRY_COUNT as f64;
    let rms_true = (true_sq_sum / E2E_ENTRY_COUNT as f64).sqrt();

    // Check unbiasedness: mean signed error should be near zero.
    let normalized_bias = mean_error.abs() / rms_true.max(1e-10);
    assert!(
        normalized_bias < E2E_BIAS_TOLERANCE,
        "Systematic bias detected: normalized mean error {normalized_bias:.4} \
         exceeds tolerance {E2E_BIAS_TOLERANCE} \
         (mean_error={mean_error:.6}, rms_true={rms_true:.6})"
    );

    // Check bounded error: mean relative error should be reasonable.
    let mean_relative_error = mean_abs_error / rms_true.max(1e-10);
    assert!(
        mean_relative_error < E2E_MAX_MEAN_RELATIVE_ERROR,
        "Mean relative error {mean_relative_error:.4} exceeds tolerance \
         {E2E_MAX_MEAN_RELATIVE_ERROR} (mean_abs_error={mean_abs_error:.6})"
    );
}

// ---------------------------------------------------------------------------
// Large-cache end-to-end quality tests (1000 entries)
// ---------------------------------------------------------------------------

/// Number of KV pairs pushed in the large-cache E2E tests.
const LARGE_E2E_ENTRY_COUNT: usize = 1_000;

/// Number of random queries to test against the large cache.
const LARGE_E2E_QUERY_COUNT: usize = 100;

/// Dimension for large E2E tests (d=128).
const LARGE_E2E_DIM_128: usize = 128;

/// Dimension for large E2E tests (d=256).
const LARGE_E2E_DIM_256: usize = 256;

/// Seed offset for large E2E key generation.
const LARGE_E2E_KEY_SEED_OFFSET: u64 = 30000;

/// Seed offset for large E2E value generation.
const LARGE_E2E_VALUE_SEED_OFFSET: u64 = 40000;

/// Seed offset for large E2E query generation.
const LARGE_E2E_QUERY_SEED_OFFSET: u64 = 50000;

/// QJL seed for large E2E tests.
const LARGE_E2E_QJL_SEED: u64 = 77777;

/// Rotation seed for large E2E tests.
const LARGE_E2E_ROTATION_SEED: u64 = 42;

/// Maximum acceptable mean relative error over 100 queries against 1000 entries.
const LARGE_E2E_MAX_MEAN_RELATIVE_ERROR: f64 = 0.5;

/// Maximum acceptable normalized bias over 100 queries.
const LARGE_E2E_BIAS_TOLERANCE: f64 = 0.08;

/// Bit budget for large E2E tests.
const LARGE_E2E_BITS: u8 = 3;

/// Runs the large-cache E2E test at a given dimension.
///
/// Creates a QuantizedKVCache with `LARGE_E2E_ENTRY_COUNT` entries, then for
/// `LARGE_E2E_QUERY_COUNT` random queries:
///   - Computes attention_scores via cache
///   - Computes true dot products with original keys
///   - Checks mean relative error
///   - Checks no systematic bias
fn run_large_cache_e2e(dim: usize) {
    let config = TurboQuantConfig::new(LARGE_E2E_BITS, dim)
        .unwrap()
        .with_seed(LARGE_E2E_ROTATION_SEED);
    let mut cache = QuantizedKVCache::new(config, 1, LARGE_E2E_QJL_SEED);

    // Store original keys for ground-truth comparison.
    let mut original_keys: Vec<Vec<f32>> = Vec::with_capacity(LARGE_E2E_ENTRY_COUNT);

    for i in 0..LARGE_E2E_ENTRY_COUNT {
        let key_seed = (i as u64)
            .wrapping_mul(LCG_MULTIPLIER)
            .wrapping_add(LARGE_E2E_KEY_SEED_OFFSET);
        let val_seed = (i as u64)
            .wrapping_mul(LCG_MULTIPLIER)
            .wrapping_add(LARGE_E2E_VALUE_SEED_OFFSET);

        let key = pseudo_random_vec(dim, key_seed);
        let val = pseudo_random_vec(dim, val_seed);

        original_keys.push(key.clone());
        cache.push(0, &key, &val).unwrap();
    }

    // Accumulate error statistics over many queries.
    let mut total_bias = 0.0_f64;
    let mut total_abs_error = 0.0_f64;
    let mut total_true_sq = 0.0_f64;
    let mut total_score_count = 0usize;

    for q in 0..LARGE_E2E_QUERY_COUNT {
        let query_seed = (q as u64)
            .wrapping_mul(LCG_MULTIPLIER)
            .wrapping_add(LARGE_E2E_QUERY_SEED_OFFSET);
        let query = pseudo_random_vec(dim, query_seed);

        let scores = cache.attention_scores(0, &query).unwrap();
        assert_eq!(scores.len(), LARGE_E2E_ENTRY_COUNT);

        for (i, &score) in scores.iter().enumerate() {
            let true_ip = dot_product(&query, &original_keys[i]) as f64;
            let error = score as f64 - true_ip;
            total_bias += error;
            total_abs_error += error.abs();
            total_true_sq += true_ip * true_ip;
            total_score_count += 1;
        }
    }

    let mean_bias = total_bias / total_score_count as f64;
    let mean_abs_error = total_abs_error / total_score_count as f64;
    let rms_true = (total_true_sq / total_score_count as f64).sqrt();

    // Check no systematic bias.
    let normalized_bias = mean_bias.abs() / rms_true.max(1e-10);
    eprintln!(
        "Large E2E d={dim}: normalized_bias={normalized_bias:.4}, \
         mean_rel_error={:.4}, rms_true={rms_true:.6}",
        mean_abs_error / rms_true.max(1e-10)
    );
    assert!(
        normalized_bias < LARGE_E2E_BIAS_TOLERANCE,
        "Large E2E d={dim}: systematic bias detected: normalized mean error \
         {normalized_bias:.4} exceeds tolerance {LARGE_E2E_BIAS_TOLERANCE} \
         (mean_bias={mean_bias:.6}, rms_true={rms_true:.6})"
    );

    // Check mean relative error.
    let mean_relative_error = mean_abs_error / rms_true.max(1e-10);
    assert!(
        mean_relative_error < LARGE_E2E_MAX_MEAN_RELATIVE_ERROR,
        "Large E2E d={dim}: mean relative error {mean_relative_error:.4} exceeds \
         tolerance {LARGE_E2E_MAX_MEAN_RELATIVE_ERROR}"
    );
}

#[test]
fn large_cache_e2e_attention_quality_d128() {
    run_large_cache_e2e(LARGE_E2E_DIM_128);
}

#[test]
fn large_cache_e2e_attention_quality_d256() {
    run_large_cache_e2e(LARGE_E2E_DIM_256);
}
