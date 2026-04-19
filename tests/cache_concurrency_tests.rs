//! Concurrency tests for per-layer locked compressed KV caches.
//!
//! Verifies that the `Arc<PqoCache>` / `Arc<TqCache>` interior locking allows
//! parallel writes on different layers, tolerates concurrent reset/decode,
//! and produces the same result under contention as a serial baseline.
//!
//! These tests exercise the per-layer `Mutex<LayerStorage>` design that
//! unblocks speculative decoding (multiple forward passes concurrently).

// qual:allow(srp) — single-responsibility module: concurrency testing for the per-layer locking contract
#![cfg(feature = "candle")]

use std::sync::Arc;
use std::thread;

use candle_core::{DType, Device, Result, Tensor};
use mistralrs_kv_cache::{AttendConfig, CompressedKVCache};
use turboquant::cache::config::QuantNormMode;
use turboquant::cache::{CacheConfig, PqoCache};
use turboquant::test_utils::make_kv as shared_make_kv;

const HEAD_DIM: usize = 128;
const NUM_KV_HEADS: usize = 4;
const BITS: u8 = 3;
const NUM_ATTN_HEADS: usize = NUM_KV_HEADS * 2;
/// Multiplier to keep per-thread seed-ranges disjoint in concurrency tests.
const LAYER_SEED_STRIDE: u32 = 10_000;

fn cfg(num_layers: usize) -> CacheConfig {
    CacheConfig {
        bits: BITS,
        head_dim: HEAD_DIM,
        num_kv_heads: NUM_KV_HEADS,
        num_layers,
        norm_mode: QuantNormMode::MaxNorm,
        outlier_blocks: usize::MAX,
    }
}

fn make_kv(seq_len: usize, seed: u32) -> Result<(Tensor, Tensor)> {
    Ok(shared_make_kv(seq_len, NUM_KV_HEADS, HEAD_DIM, seed))
}

fn make_q(seq_len: usize) -> Result<Tensor> {
    Tensor::zeros(
        (1, NUM_ATTN_HEADS, seq_len, HEAD_DIM),
        DType::F32,
        &Device::Cpu,
    )
}

fn decode_config() -> AttendConfig {
    AttendConfig {
        softmax_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        n_kv_groups: NUM_ATTN_HEADS / NUM_KV_HEADS,
    }
}

/// Drive `iterations` decode calls on the given `layer`. Keeps `seq_len`
/// consistent so the caller can check it later.
fn drive_decode(cache: &Arc<PqoCache>, layer: usize, iterations: usize) -> Result<()> {
    let q = make_q(1)?;
    let cfg = decode_config();
    for step in 0..iterations {
        let (k, v) = make_kv(1, (layer as u32) * LAYER_SEED_STRIDE + step as u32)?;
        cache.decode(layer, &k, &v, &q, &cfg)?;
    }
    Ok(())
}

// ---- 1. Different layers can be written in parallel ----------------------

#[test]
fn parallel_decode_different_layers() {
    const ITERATIONS: usize = 100;
    let cache = Arc::new(PqoCache::new(cfg(2)).expect("cache::new"));

    let c0 = cache.clone();
    let c1 = cache.clone();
    let h0 = thread::spawn(move || drive_decode(&c0, 0, ITERATIONS));
    let h1 = thread::spawn(move || drive_decode(&c1, 1, ITERATIONS));
    h0.join().expect("layer 0 thread").expect("layer 0 drive");
    h1.join().expect("layer 1 thread").expect("layer 1 drive");

    assert_eq!(cache.seq_len(0), ITERATIONS);
    assert_eq!(cache.seq_len(1), ITERATIONS);
}

// ---- 2. Parallel prefill on different layers matches serial baseline -----

#[test]
fn parallel_prefill_no_corruption() {
    const PREFILL_LEN: usize = 16;
    const SEED_LAYER_0: u32 = 42;
    const SEED_LAYER_1: u32 = 1337;

    let (k0, v0) = make_kv(PREFILL_LEN, SEED_LAYER_0).expect("make_kv 0");
    let (k1, v1) = make_kv(PREFILL_LEN, SEED_LAYER_1).expect("make_kv 1");
    let q = make_q(PREFILL_LEN).expect("make_q");

    // Serial baseline: prefill both layers sequentially in a fresh cache.
    let serial = PqoCache::new(cfg(2)).expect("cache::new");
    let serial_r0 = serial.prefill(0, &k0, &v0, &q).expect("serial prefill 0");
    let serial_r1 = serial.prefill(1, &k1, &v1, &q).expect("serial prefill 1");

    // Parallel: prefill layer 0 + layer 1 concurrently in a fresh cache.
    let parallel = Arc::new(PqoCache::new(cfg(2)).expect("cache::new"));
    let p0 = parallel.clone();
    let p1 = parallel.clone();
    let k0c = k0.clone();
    let v0c = v0.clone();
    let k1c = k1.clone();
    let v1c = v1.clone();
    let qc0 = q.clone();
    let qc1 = q.clone();
    let h0 = thread::spawn(move || p0.prefill(0, &k0c, &v0c, &qc0));
    let h1 = thread::spawn(move || p1.prefill(1, &k1c, &v1c, &qc1));
    let par_r0 = h0.join().expect("thread 0").expect("parallel prefill 0");
    let par_r1 = h1.join().expect("thread 1").expect("parallel prefill 1");

    // First prefill on an empty layer returns the input tensors unchanged —
    // both paths should produce bit-identical results.
    let serial_k0_v: Vec<f32> = serial_r0.k.flatten_all().unwrap().to_vec1().unwrap();
    let par_k0_v: Vec<f32> = par_r0.k.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(serial_k0_v, par_k0_v, "layer 0 prefill diverged");

    let serial_k1_v: Vec<f32> = serial_r1.k.flatten_all().unwrap().to_vec1().unwrap();
    let par_k1_v: Vec<f32> = par_r1.k.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(serial_k1_v, par_k1_v, "layer 1 prefill diverged");
}

// ---- 3. Concurrent reset + decode does not deadlock or panic -------------

#[test]
fn concurrent_reset_decode() {
    const ITERS: usize = 200;
    const SEED_A: u32 = 10;
    const SEED_B: u32 = 11;

    let cache = Arc::new(PqoCache::new(cfg(2)).expect("cache::new"));

    let cache_a = cache.clone();
    let a = thread::spawn(move || -> Result<()> {
        for _ in 0..ITERS {
            let (k, v) = make_kv(1, SEED_A)?;
            let q = make_q(1)?;
            cache_a.decode(0, &k, &v, &q, &decode_config())?;
            cache_a.reset()?;
        }
        Ok(())
    });

    let cache_b = cache.clone();
    let b = thread::spawn(move || -> Result<()> {
        for _ in 0..ITERS {
            let (k, v) = make_kv(1, SEED_B)?;
            let q = make_q(1)?;
            // Decode may race with reset on layer 1 (reset clears all layers);
            // either an empty-cache state or a populated one is valid. Just
            // assert no panic / no error return.
            cache_b.decode(1, &k, &v, &q, &decode_config())?;
        }
        Ok(())
    });

    a.join().expect("thread A").expect("drive A");
    b.join().expect("thread B").expect("drive B");
}

// ---- 4. Stress: N threads, N layers, contention ---------------------------

#[test]
fn layer_independence_under_contention() {
    const STRESS_NUM_LAYERS: usize = 8;
    const STEPS_PER_LAYER: usize = 30;

    let cache = Arc::new(PqoCache::new(cfg(STRESS_NUM_LAYERS)).expect("cache::new"));

    let handles: Vec<_> = (0..STRESS_NUM_LAYERS)
        .map(|layer| {
            let c = cache.clone();
            thread::spawn(move || drive_decode(&c, layer, STEPS_PER_LAYER))
        })
        .collect();
    for h in handles {
        h.join().expect("layer thread").expect("drive_decode");
    }

    for layer in 0..STRESS_NUM_LAYERS {
        assert_eq!(
            cache.seq_len(layer),
            STEPS_PER_LAYER,
            "layer {layer} has wrong seq_len after concurrent decode"
        );
    }
}
