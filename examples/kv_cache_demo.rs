//! Demonstrates using TurboQuant as a quantized KV cache with attention.
//!
//! Creates a single-layer cache, pushes 1024 key-value pairs, computes
//! attention scores for a query, and reports memory statistics.
//!
//! No external dependencies (no `rand`). Uses deterministic LCG + sine
//! patterns for reproducible data.

use turboquant::attention::QuantizedKVCache;
use turboquant::packed::TurboQuantConfig;

// ---------------------------------------------------------------------------
// Named constants
// ---------------------------------------------------------------------------

/// Number of quantization bits (TQ3).
const BITS: u8 = 3;

/// Head dimension (typical for LLMs like LLaMA, Mistral).
const DIM: usize = 128;

/// Seed for the PolarQuant rotation sign pattern.
const ROTATION_SEED: u64 = 42;

/// Seed for the QJL Rademacher matrix.
const QJL_SEED: u64 = 12345;

/// Number of layers in the cache.
const NUM_LAYERS: usize = 1;

/// Layer index for this demo.
const LAYER: usize = 0;

/// Number of KV pairs to push into the cache.
const NUM_ENTRIES: usize = 1024;

/// Number of attention scores to display.
const DISPLAY_SCORES: usize = 8;

/// LCG multiplier (Knuth's constant).
const LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;

/// LCG increment.
const LCG_INCREMENT: u64 = 1;

/// Right-shift for extracting bits from LCG state.
const LCG_SHIFT: u32 = 33;

/// Amplitude for key vector generation.
const KEY_AMPLITUDE: f32 = 1.0;

/// Amplitude for value vector generation.
const VALUE_AMPLITUDE: f32 = 0.5;

/// Query frequency scaling.
const QUERY_FREQUENCY: f64 = 0.25;

/// Number of bytes per FP16 element.
const BYTES_PER_FP16: usize = 2;

/// Number of KV components (key + value).
const KV_PAIR_COUNT: usize = 2;

/// Bytes per kilobyte.
const BYTES_PER_KB: f64 = 1024.0;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Deterministic pseudo-random vector using a simple LCG.
fn lcg_vec(dim: usize, seed: u64, amplitude: f32) -> Vec<f32> {
    let mut state = seed;
    (0..dim)
        .map(|_| {
            state = state
                .wrapping_mul(LCG_MULTIPLIER)
                .wrapping_add(LCG_INCREMENT);
            let bits = (state >> LCG_SHIFT) as i32;
            amplitude * (bits as f32 / i32::MAX as f32)
        })
        .collect()
}

fn main() {
    // -- 1. Create cache ------------------------------------------------------
    let config = TurboQuantConfig::new(BITS, DIM)
        .expect("valid config")
        .with_seed(ROTATION_SEED);

    let mut cache = QuantizedKVCache::new(config, NUM_LAYERS, QJL_SEED);

    // -- 2. Push KV pairs (batch) -----------------------------------------------
    println!("Pushing {NUM_ENTRIES} key-value pairs via push_batch (d={DIM}, TQ{BITS})...");

    let keys: Vec<Vec<f32>> = (0..NUM_ENTRIES)
        .map(|i| {
            let seed = (i as u64).wrapping_mul(LCG_MULTIPLIER).wrapping_add(1000);
            lcg_vec(DIM, seed, KEY_AMPLITUDE)
        })
        .collect();
    let values: Vec<Vec<f32>> = (0..NUM_ENTRIES)
        .map(|i| {
            let seed = (i as u64).wrapping_mul(LCG_MULTIPLIER).wrapping_add(2000);
            lcg_vec(DIM, seed, VALUE_AMPLITUDE)
        })
        .collect();

    let key_refs: Vec<&[f32]> = keys.iter().map(|k| k.as_slice()).collect();
    let val_refs: Vec<&[f32]> = values.iter().map(|v| v.as_slice()).collect();

    cache
        .push_batch(LAYER, &key_refs, &val_refs)
        .expect("push_batch succeeded");

    // -- 3. Compute attention scores ------------------------------------------
    let query: Vec<f32> = (0..DIM)
        .map(|i| (i as f64 * QUERY_FREQUENCY).sin() as f32)
        .collect();

    let scores = cache
        .attention_scores(LAYER, &query)
        .expect("attention succeeded");

    // -- 4. Print cache stats -------------------------------------------------
    let quantized_bytes = cache.memory_usage();
    let fp16_bytes = cache.fp16_equivalent_memory();
    let compression_ratio = fp16_bytes as f64 / quantized_bytes as f64;

    println!();
    println!("Cache Statistics");
    println!("================");
    println!("Entries:            {}", cache.entry_count(LAYER));
    println!(
        "Quantized memory:   {:.1} KB",
        quantized_bytes as f64 / BYTES_PER_KB
    );
    println!(
        "FP16 equivalent:    {:.1} KB",
        fp16_bytes as f64 / BYTES_PER_KB
    );
    println!("Compression ratio:  {compression_ratio:.2}x");
    println!();

    // Sanity check: FP16 equivalent should match manual calculation
    let expected_fp16 = NUM_ENTRIES * DIM * BYTES_PER_FP16 * KV_PAIR_COUNT;
    assert_eq!(fp16_bytes, expected_fp16);

    // -- 5. Print attention scores --------------------------------------------
    println!(
        "Attention scores (first {DISPLAY_SCORES} of {}):",
        scores.len()
    );
    for (i, &score) in scores.iter().take(DISPLAY_SCORES).enumerate() {
        println!("  key[{i:4}]: {score:+.6}");
    }
}
