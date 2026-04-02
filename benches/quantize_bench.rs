//! Criterion benchmarks for TurboQuant quantization, dequantization,
//! QJL inner-product estimation, and attention operations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use turboquant::attention::QuantizedKVCache;
use turboquant::packed::TurboQuantConfig;
use turboquant::qjl::{estimate_inner_product, precompute_query_projections, quantize_with_qjl};
use turboquant::quantize::{dequantize_vec, quantize_vec};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DIM_64: usize = 64;
const DIM_128: usize = 128;
const DIM_256: usize = 256;

const BITS_TQ3: u8 = 3;
const BITS_TQ4: u8 = 4;

const ROTATION_SEED: u64 = 42;
const QJL_SEED: u64 = 12345;
const LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;
const LCG_INCREMENT: u64 = 1;
const LCG_SHIFT: u32 = 33;

const CACHE_SEQ_LEN: usize = 1024;
const BENCH_NUM_LAYERS: usize = 1;
const BENCH_LAYER: usize = 0;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

fn make_config(bits: u8, dim: usize) -> TurboQuantConfig {
    TurboQuantConfig::new(bits, dim)
        .unwrap()
        .with_seed(ROTATION_SEED)
}

// ---------------------------------------------------------------------------
// Benchmark: quantize_vec
// ---------------------------------------------------------------------------

fn bench_quantize(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantize_vec");

    for &(bits, dim) in &[
        (BITS_TQ3, DIM_64),
        (BITS_TQ3, DIM_128),
        (BITS_TQ3, DIM_256),
        (BITS_TQ4, DIM_128),
    ] {
        let config = make_config(bits, dim);
        let data = pseudo_random_vec(dim, 1000);
        let label = format!("tq{bits}_d{dim}");

        group.bench_with_input(BenchmarkId::new("polarquant", &label), &data, |b, data| {
            b.iter(|| quantize_vec(black_box(&config), black_box(data)))
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: dequantize_vec
// ---------------------------------------------------------------------------

fn bench_dequantize(c: &mut Criterion) {
    let mut group = c.benchmark_group("dequantize_vec");

    for &(bits, dim) in &[
        (BITS_TQ3, DIM_64),
        (BITS_TQ3, DIM_128),
        (BITS_TQ3, DIM_256),
        (BITS_TQ4, DIM_128),
    ] {
        let config = make_config(bits, dim);
        let data = pseudo_random_vec(dim, 2000);
        let block = quantize_vec(&config, &data).unwrap();
        let label = format!("tq{bits}_d{dim}");

        group.bench_with_input(
            BenchmarkId::new("polarquant", &label),
            &block,
            |b, block| b.iter(|| dequantize_vec(black_box(&config), black_box(block))),
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: quantize_with_qjl
// ---------------------------------------------------------------------------

fn bench_quantize_qjl(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantize_with_qjl");

    for &(bits, dim) in &[(BITS_TQ3, DIM_128), (BITS_TQ4, DIM_128)] {
        let config = make_config(bits, dim);
        let data = pseudo_random_vec(dim, 3000);
        let label = format!("tq{bits}_d{dim}");

        group.bench_with_input(BenchmarkId::new("qjl", &label), &data, |b, data| {
            b.iter(|| quantize_with_qjl(black_box(&config), black_box(data), QJL_SEED))
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: estimate_inner_product (pre-computed projections + SIMD)
// ---------------------------------------------------------------------------

fn bench_inner_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("estimate_inner_product");

    for &(bits, dim) in &[(BITS_TQ3, DIM_128), (BITS_TQ4, DIM_128)] {
        let config = make_config(bits, dim);
        let key = pseudo_random_vec(dim, 4000);
        let query = pseudo_random_vec(dim, 5000);
        let block = quantize_with_qjl(&config, &key, QJL_SEED).unwrap();
        let r_query = precompute_query_projections(&query, dim, QJL_SEED);
        let label = format!("tq{bits}_d{dim}");

        group.bench_with_input(
            BenchmarkId::new("qjl", &label),
            &(&query, &r_query, &block),
            |b, &(query, r_query, block)| {
                b.iter(|| {
                    estimate_inner_product(
                        black_box(query),
                        black_box(r_query),
                        black_box(block),
                        &config,
                    )
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: precompute_query_projections
// ---------------------------------------------------------------------------

fn bench_precompute_projections(c: &mut Criterion) {
    let mut group = c.benchmark_group("precompute_query_projections");

    for &dim in &[DIM_64, DIM_128, DIM_256] {
        let query = pseudo_random_vec(dim, 5000);
        let label = format!("d{dim}");

        group.bench_with_input(
            BenchmarkId::new("hash_rademacher", &label),
            &query,
            |b, query| b.iter(|| precompute_query_projections(black_box(query), dim, QJL_SEED)),
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: attention_scores over 1024 keys
// ---------------------------------------------------------------------------

fn bench_attention_scores(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_scores_1024");

    let config = make_config(BITS_TQ3, DIM_128);
    let mut cache = QuantizedKVCache::new(config, BENCH_NUM_LAYERS, QJL_SEED);

    for i in 0..CACHE_SEQ_LEN {
        let key = pseudo_random_vec(DIM_128, i as u64 * 100);
        let value = pseudo_random_vec(DIM_128, i as u64 * 100 + 50);
        let _ = cache.push(BENCH_LAYER, &key, &value);
    }

    let query = pseudo_random_vec(DIM_128, 99999);

    group.bench_function("tq3_d128", |b| {
        b.iter(|| cache.attention_scores(black_box(BENCH_LAYER), black_box(&query)))
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: memory comparison
// ---------------------------------------------------------------------------

fn bench_memory_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_comparison");

    let config = make_config(BITS_TQ3, DIM_128);
    let mut cache = QuantizedKVCache::new(config, BENCH_NUM_LAYERS, QJL_SEED);

    let entry_counts = [64, 256, 1024];

    for &count in &entry_counts {
        // Fill cache to desired count (reuse previous entries)
        while cache.entry_count(BENCH_LAYER) < count {
            let i = cache.entry_count(BENCH_LAYER);
            let key = pseudo_random_vec(DIM_128, i as u64 * 200);
            let value = pseudo_random_vec(DIM_128, i as u64 * 200 + 100);
            let _ = cache.push(BENCH_LAYER, &key, &value);
        }

        let tq3_bytes = cache.memory_usage();
        let fp16_bytes = cache.fp16_equivalent_memory();
        let ratio = fp16_bytes as f64 / tq3_bytes as f64;

        group.bench_function(BenchmarkId::new("memory_usage", count), |b| {
            b.iter(|| black_box(cache.memory_usage()))
        });

        // Print compression ratio (visible in benchmark output)
        println!(
            "  entries={count}: TQ3={tq3_bytes} bytes, FP16={fp16_bytes} bytes, compression={ratio:.1}x"
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Groups
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_quantize,
    bench_dequantize,
    bench_quantize_qjl,
    bench_inner_product,
    bench_precompute_projections,
    bench_attention_scores,
    bench_memory_comparison,
);
criterion_main!(benches);
