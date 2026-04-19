//! QJL inner-product estimator: bias + variance statistical checks.
//!
//! Extracted from the former `inner_product_tests.rs`.

use turboquant::packed::TurboQuantConfig;
use turboquant::qjl::{dot_product, estimate_inner_product_single, quantize_with_qjl};
use turboquant::test_utils::{pseudo_random_vec, LCG_MULTIPLIER};

const TEST_DIM: usize = 64;
const ROTATION_SEED: u64 = 42;
const QJL_SEED: u64 = 12345;
const BITS_3: u8 = 3;
const KEY_SEED_OFFSET: u64 = 1000;
const QUERY_SEED_OFFSET: u64 = 2000;

const LARGE_SAMPLE_COUNT: usize = 10_000;
const QUICK_SAMPLE_COUNT: usize = 200;

const LARGE_BIAS_TOLERANCE: f32 = 0.02;
const QUICK_BIAS_TOLERANCE: f32 = 0.1;
const MAX_RELATIVE_VARIANCE: f64 = 2.0;

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
        mean_bias < LARGE_BIAS_TOLERANCE,
        "mean bias {mean_bias} exceeds tolerance {LARGE_BIAS_TOLERANCE} over {LARGE_SAMPLE_COUNT} pairs"
    );
}

#[test]
fn qjl_inner_product_bias_and_variance_quick() {
    let config = TurboQuantConfig::new(BITS_3, TEST_DIM)
        .unwrap()
        .with_seed(ROTATION_SEED);
    let mut samples: Vec<(f64, f64)> = Vec::with_capacity(QUICK_SAMPLE_COUNT);
    for i in 0..QUICK_SAMPLE_COUNT {
        let key_seed = (i as u64)
            .wrapping_mul(LCG_MULTIPLIER)
            .wrapping_add(KEY_SEED_OFFSET);
        let query_seed = (i as u64)
            .wrapping_mul(LCG_MULTIPLIER)
            .wrapping_add(QUERY_SEED_OFFSET);
        let qjl_seed = QJL_SEED.wrapping_add(i as u64);

        let key = pseudo_random_vec(TEST_DIM, key_seed);
        let query = pseudo_random_vec(TEST_DIM, query_seed);
        let true_ip = dot_product(&key, &query) as f64;
        let block = quantize_with_qjl(&config, &key, qjl_seed).unwrap();
        let est = estimate_inner_product_single(&query, &block, &config, qjl_seed).unwrap() as f64;
        samples.push((true_ip, est));
    }

    let n = samples.len();
    let bias_sum: f64 = samples.iter().map(|(t, e)| e - t).sum();
    let mean_bias = (bias_sum / n as f64).abs() as f32;
    assert!(
        mean_bias < QUICK_BIAS_TOLERANCE,
        "mean bias {mean_bias} exceeds tolerance {QUICK_BIAS_TOLERANCE} over {n} pairs"
    );

    let sum_sq_error: f64 = samples.iter().map(|(t, e)| (e - t).powi(2)).sum();
    let sum_true_sq: f64 = samples.iter().map(|(t, _)| t * t).sum();
    let mean_sq_error = sum_sq_error / n as f64;
    let mean_true_sq = sum_true_sq / n as f64;
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
