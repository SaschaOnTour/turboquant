//! Compression-ratio tests for `QuantizedKVCache` at multiple (bits, dim).
//!
//! Extracted from the former `mse_validation.rs`.

use turboquant::packed::TurboQuantConfig;
use turboquant::test_utils::pseudo_random_vec;
use turboquant::QuantizedKVCache;

/// Rotation seed (shared across MSE tests).
const MSE_SEED: u64 = 42;

/// Number of entries for compression ratio tests.
const COMPRESSION_ENTRY_COUNT: usize = 10;
/// QJL seed for compression tests.
const COMPRESSION_QJL_SEED: u64 = 99999;
/// Seed offset between entries.
const COMPRESSION_SEED_OFFSET: u64 = 500;

/// Minimum expected compression ratios by configuration.
const TQ3_D128_MIN_COMPRESSION: f32 = 4.0;
const TQ3_D256_MIN_COMPRESSION: f32 = 4.5;
const TQ4_D128_MIN_COMPRESSION: f32 = 3.0;
const TQ4_D256_MIN_COMPRESSION: f32 = 3.5;

fn measure_compression_ratio(bits: u8, dim: usize) -> f32 {
    let config = TurboQuantConfig::new(bits, dim)
        .unwrap()
        .with_seed(MSE_SEED);
    let mut cache = QuantizedKVCache::new(config, 1, COMPRESSION_QJL_SEED);

    for i in 0..COMPRESSION_ENTRY_COUNT {
        let key = pseudo_random_vec(dim, 10000 + i as u64 * COMPRESSION_SEED_OFFSET);
        let val = pseudo_random_vec(dim, 20000 + i as u64 * COMPRESSION_SEED_OFFSET);
        cache.push(0, &key, &val).unwrap();
    }

    let quantized_bytes = cache.memory_usage();
    let fp16_bytes = cache.fp16_equivalent_memory();
    fp16_bytes as f32 / quantized_bytes as f32
}

#[test]
fn compression_ratio_tq3_d128() {
    let ratio = measure_compression_ratio(3, 128);
    eprintln!("TQ3 d=128 compression ratio: {ratio:.2}x");
    assert!(
        ratio >= TQ3_D128_MIN_COMPRESSION,
        "TQ3 d=128 compression ratio {ratio:.2}x below minimum {TQ3_D128_MIN_COMPRESSION}x"
    );
}

#[test]
fn compression_ratio_tq3_d256() {
    let ratio = measure_compression_ratio(3, 256);
    eprintln!("TQ3 d=256 compression ratio: {ratio:.2}x");
    assert!(
        ratio >= TQ3_D256_MIN_COMPRESSION,
        "TQ3 d=256 compression ratio {ratio:.2}x below minimum {TQ3_D256_MIN_COMPRESSION}x"
    );
}

#[test]
fn compression_ratio_tq4_d128() {
    let ratio = measure_compression_ratio(4, 128);
    eprintln!("TQ4 d=128 compression ratio: {ratio:.2}x");
    assert!(
        ratio >= TQ4_D128_MIN_COMPRESSION,
        "TQ4 d=128 compression ratio {ratio:.2}x below minimum {TQ4_D128_MIN_COMPRESSION}x"
    );
}

#[test]
fn compression_ratio_tq4_d256() {
    let ratio = measure_compression_ratio(4, 256);
    eprintln!("TQ4 d=256 compression ratio: {ratio:.2}x");
    assert!(
        ratio >= TQ4_D256_MIN_COMPRESSION,
        "TQ4 d=256 compression ratio {ratio:.2}x below minimum {TQ4_D256_MIN_COMPRESSION}x"
    );
}
