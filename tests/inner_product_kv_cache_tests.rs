//! End-to-end QJL attention-score quality tests via `QuantizedKVCache`.
//!
//! Extracted from the former `inner_product_tests.rs`.

use turboquant::packed::TurboQuantConfig;
use turboquant::qjl::dot_product;
use turboquant::test_utils::{pseudo_random_vec, LCG_MULTIPLIER};
use turboquant::QuantizedKVCache;

const BITS_3: u8 = 3;
const ROTATION_SEED: u64 = 42;

// ---- small E2E (100 entries, single query) ----

const SMALL_E2E_ENTRY_COUNT: usize = 100;
const SMALL_E2E_DIM: usize = 128;
const SMALL_E2E_KEY_OFFSET: u64 = 5000;
const SMALL_E2E_VAL_OFFSET: u64 = 6000;
const SMALL_E2E_QUERY_OFFSET: u64 = 8000;
const SMALL_E2E_QJL_SEED: u64 = 54321;
const SMALL_E2E_BIAS_TOLERANCE: f64 = 0.05;
const SMALL_E2E_MAX_RELATIVE_ERROR: f64 = 0.5;

// ---- large E2E (1000 entries × 100 queries) ----

const LARGE_E2E_ENTRY_COUNT: usize = 1_000;
const LARGE_E2E_QUERY_COUNT: usize = 100;
const LARGE_E2E_KEY_OFFSET: u64 = 30000;
const LARGE_E2E_VAL_OFFSET: u64 = 40000;
const LARGE_E2E_QUERY_OFFSET: u64 = 50000;
const LARGE_E2E_QJL_SEED: u64 = 77777;
const LARGE_E2E_BIAS_TOLERANCE: f64 = 0.08;
const LARGE_E2E_MAX_RELATIVE_ERROR: f64 = 0.5;

/// Running score-error totals; one per test run, consumed into `ScoreStats`.
#[derive(Default)]
struct ScoreAccumulator {
    bias_sum: f64,
    abs_error_sum: f64,
    true_sq_sum: f64,
    count: usize,
}

impl ScoreAccumulator {
    fn add_batch(&mut self, scores: &[f32], query: &[f32], original_keys: &[Vec<f32>]) {
        for (i, &score) in scores.iter().enumerate() {
            let true_ip = dot_product(query, &original_keys[i]) as f64;
            let error = score as f64 - true_ip;
            self.bias_sum += error;
            self.abs_error_sum += error.abs();
            self.true_sq_sum += true_ip * true_ip;
        }
        self.count += scores.len();
    }

    fn finish(&self) -> ScoreStats {
        let n = self.count as f64;
        ScoreStats {
            mean_bias: self.bias_sum / n,
            mean_abs_error: self.abs_error_sum / n,
            rms_true: (self.true_sq_sum / n).sqrt(),
        }
    }
}

/// Aggregated statistics for a run of score comparisons.
struct ScoreStats {
    mean_bias: f64,
    mean_abs_error: f64,
    rms_true: f64,
}

impl ScoreStats {
    fn normalized_bias(&self) -> f64 {
        self.mean_bias.abs() / self.rms_true.max(1e-10)
    }

    fn mean_relative_error(&self) -> f64 {
        self.mean_abs_error / self.rms_true.max(1e-10)
    }
}

#[test]
fn e2e_kv_cache_attention_scores_unbiased() {
    let config = TurboQuantConfig::new(BITS_3, SMALL_E2E_DIM)
        .unwrap()
        .with_seed(ROTATION_SEED);
    let mut cache = QuantizedKVCache::new(config, 1, SMALL_E2E_QJL_SEED);
    let mut original_keys: Vec<Vec<f32>> = Vec::with_capacity(SMALL_E2E_ENTRY_COUNT);
    for i in 0..SMALL_E2E_ENTRY_COUNT {
        let key_seed = (i as u64)
            .wrapping_mul(LCG_MULTIPLIER)
            .wrapping_add(SMALL_E2E_KEY_OFFSET);
        let val_seed = (i as u64)
            .wrapping_mul(LCG_MULTIPLIER)
            .wrapping_add(SMALL_E2E_VAL_OFFSET);
        let key = pseudo_random_vec(SMALL_E2E_DIM, key_seed);
        let val = pseudo_random_vec(SMALL_E2E_DIM, val_seed);
        original_keys.push(key.clone());
        cache.push(0, &key, &val).unwrap();
    }

    let query = pseudo_random_vec(SMALL_E2E_DIM, SMALL_E2E_QUERY_OFFSET);
    let scores = cache.attention_scores(0, &query).unwrap();
    assert_eq!(scores.len(), SMALL_E2E_ENTRY_COUNT);

    let mut acc = ScoreAccumulator::default();
    acc.add_batch(&scores, &query, &original_keys);
    let stats = acc.finish();

    assert!(
        stats.normalized_bias() < SMALL_E2E_BIAS_TOLERANCE,
        "Systematic bias detected: normalized mean error {:.4} exceeds tolerance {SMALL_E2E_BIAS_TOLERANCE}",
        stats.normalized_bias()
    );
    assert!(
        stats.mean_relative_error() < SMALL_E2E_MAX_RELATIVE_ERROR,
        "Mean relative error {:.4} exceeds tolerance {SMALL_E2E_MAX_RELATIVE_ERROR}",
        stats.mean_relative_error()
    );
}

fn run_large_cache_e2e(dim: usize) {
    let config = TurboQuantConfig::new(BITS_3, dim)
        .unwrap()
        .with_seed(ROTATION_SEED);
    let mut cache = QuantizedKVCache::new(config, 1, LARGE_E2E_QJL_SEED);
    let mut original_keys: Vec<Vec<f32>> = Vec::with_capacity(LARGE_E2E_ENTRY_COUNT);
    for i in 0..LARGE_E2E_ENTRY_COUNT {
        let key_seed = (i as u64)
            .wrapping_mul(LCG_MULTIPLIER)
            .wrapping_add(LARGE_E2E_KEY_OFFSET);
        let val_seed = (i as u64)
            .wrapping_mul(LCG_MULTIPLIER)
            .wrapping_add(LARGE_E2E_VAL_OFFSET);
        let key = pseudo_random_vec(dim, key_seed);
        let val = pseudo_random_vec(dim, val_seed);
        original_keys.push(key.clone());
        cache.push(0, &key, &val).unwrap();
    }

    let mut acc = ScoreAccumulator::default();
    for q in 0..LARGE_E2E_QUERY_COUNT {
        let query_seed = (q as u64)
            .wrapping_mul(LCG_MULTIPLIER)
            .wrapping_add(LARGE_E2E_QUERY_OFFSET);
        let query = pseudo_random_vec(dim, query_seed);

        let scores = cache.attention_scores(0, &query).unwrap();
        assert_eq!(scores.len(), LARGE_E2E_ENTRY_COUNT);
        acc.add_batch(&scores, &query, &original_keys);
    }

    let stats = acc.finish();

    eprintln!(
        "Large E2E d={dim}: normalized_bias={:.4}, mean_rel_error={:.4}, rms_true={:.6}",
        stats.normalized_bias(),
        stats.mean_relative_error(),
        stats.rms_true
    );

    assert!(
        stats.normalized_bias() < LARGE_E2E_BIAS_TOLERANCE,
        "Large E2E d={dim}: systematic bias detected: normalized mean error {:.4} exceeds tolerance {LARGE_E2E_BIAS_TOLERANCE}",
        stats.normalized_bias()
    );
    assert!(
        stats.mean_relative_error() < LARGE_E2E_MAX_RELATIVE_ERROR,
        "Large E2E d={dim}: mean relative error {:.4} exceeds tolerance {LARGE_E2E_MAX_RELATIVE_ERROR}",
        stats.mean_relative_error()
    );
}

#[test]
fn large_cache_e2e_attention_quality_d128() {
    run_large_cache_e2e(128);
}

#[test]
fn large_cache_e2e_attention_quality_d256() {
    run_large_cache_e2e(256);
}
