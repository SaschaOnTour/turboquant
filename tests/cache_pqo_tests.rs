//! Integration tests for PqoCache — the primary CompressedKVCache implementation.
//!
//! CPU tests run with `cargo nextest run --features candle`.
//! GPU tests (behind `#[cfg(feature = "cuda")]`) require `--features cuda`.

#![cfg(feature = "candle")]

use candle_core::{DType, Device, Tensor};
use mistralrs_kv_cache::{AttendConfig, CompressedKVCache, DecodeOutput};
use turboquant::cache::config::QuantNormMode;
use turboquant::cache::{CacheConfig, PqoCache};

const HEAD_DIM: usize = 128;
const NUM_KV_HEADS: usize = 8;
const NUM_LAYERS: usize = 2;
const BITS: u8 = 3; // PQO3
const TEST_LAYER: usize = 0;

fn pqo_config(bits: u8, norm_mode: QuantNormMode) -> CacheConfig {
    CacheConfig {
        bits,
        head_dim: HEAD_DIM,
        num_kv_heads: NUM_KV_HEADS,
        num_layers: NUM_LAYERS,
        norm_mode,
        outlier_blocks: usize::MAX,
    }
}

/// Generate deterministic test data: [1, num_kv_heads, seq_len, head_dim]
fn make_kv(seq_len: usize, seed: f32) -> (Tensor, Tensor) {
    let device = Device::Cpu;
    let n = NUM_KV_HEADS * seq_len * HEAD_DIM;
    let k_data: Vec<f32> = (0..n)
        .map(|i| ((i as f32 + seed) * 0.0137).sin() * 2.0)
        .collect();
    let v_data: Vec<f32> = (0..n)
        .map(|i| ((i as f32 + seed + 1000.0) * 0.0213).cos() * 1.5)
        .collect();
    let k = Tensor::from_vec(k_data, (1, NUM_KV_HEADS, seq_len, HEAD_DIM), &device).unwrap();
    let v = Tensor::from_vec(v_data, (1, NUM_KV_HEADS, seq_len, HEAD_DIM), &device).unwrap();
    (k, v)
}

/// Dummy query tensor (needed by trait but unused by PQO).
fn make_q(seq_len: usize) -> Tensor {
    let num_attn_heads = NUM_KV_HEADS * 2; // GQA: 2 query heads per KV head
    Tensor::zeros(
        (1, num_attn_heads, seq_len, HEAD_DIM),
        DType::F32,
        &Device::Cpu,
    )
    .unwrap()
}

/// Cosine similarity between two tensors (flattened).
fn cosine_sim(a: &Tensor, b: &Tensor) -> f32 {
    let a_flat: Vec<f32> = a
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let b_flat: Vec<f32> = b
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let dot: f32 = a_flat.iter().zip(b_flat.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a_flat.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b_flat.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

// -----------------------------------------------------------------------
// Basic lifecycle tests
// -----------------------------------------------------------------------

#[test]
fn pqo3_prefill_returns_original_on_first_call() {
    let mut cache = PqoCache::new(pqo_config(BITS, QuantNormMode::MaxNorm));
    let (k, v) = make_kv(8, 1.0);
    let q = make_q(8);

    let result = cache.prefill(TEST_LAYER, &k, &v, &q).unwrap();

    // First prefill returns originals (no old cache to dequant)
    assert_eq!(result.k.dims(), k.dims());
    assert_eq!(result.v.dims(), v.dims());
    // Should be identical tensors
    let sim = cosine_sim(&result.k, &k);
    assert!(
        sim > 0.999,
        "First prefill should return originals, got cosine_sim={sim}"
    );
    assert!(result.logit_bias.is_none(), "PQO should have no logit_bias");
}

#[test]
fn pqo3_prefill_updates_seq_len() {
    let mut cache = PqoCache::new(pqo_config(BITS, QuantNormMode::MaxNorm));
    let (k, v) = make_kv(16, 2.0);
    let q = make_q(16);

    assert_eq!(cache.seq_len(TEST_LAYER), 0);
    cache.prefill(TEST_LAYER, &k, &v, &q).unwrap();
    assert_eq!(cache.seq_len(TEST_LAYER), 16);
}

#[test]
fn pqo3_decode_returns_dequantized() {
    let mut cache = PqoCache::new(pqo_config(BITS, QuantNormMode::MaxNorm));

    // Prefill 8 tokens
    let (k_pre, v_pre) = make_kv(8, 3.0);
    let q_pre = make_q(8);
    cache.prefill(TEST_LAYER, &k_pre, &v_pre, &q_pre).unwrap();

    // Decode 1 token
    let (k_dec, v_dec) = make_kv(1, 4.0);
    let q_dec = make_q(1);
    let config = AttendConfig {
        softmax_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        n_kv_groups: 2,
    };
    let output = cache
        .decode(TEST_LAYER, &k_dec, &v_dec, &q_dec, &config)
        .unwrap();

    match output {
        DecodeOutput::Dequantized(result) => {
            // Should contain all 9 tokens (8 prefill + 1 decode)
            assert_eq!(result.k.dims(), &[1, NUM_KV_HEADS, 9, HEAD_DIM]);
            assert_eq!(result.v.dims(), &[1, NUM_KV_HEADS, 9, HEAD_DIM]);
            assert!(result.logit_bias.is_none());
        }
        DecodeOutput::Fused(_) => panic!("CPU should not use fused path"),
    }
    assert_eq!(cache.seq_len(TEST_LAYER), 9);
}

#[test]
fn pqo3_roundtrip_quality_maxnorm() {
    let mut cache = PqoCache::new(pqo_config(BITS, QuantNormMode::MaxNorm));

    // Prefill 4 tokens, then decode 1 token
    let (k_pre, v_pre) = make_kv(4, 5.0);
    let q = make_q(4);
    cache.prefill(TEST_LAYER, &k_pre, &v_pre, &q).unwrap();

    let (k_dec, v_dec) = make_kv(1, 6.0);
    let q_dec = make_q(1);
    let config = AttendConfig {
        softmax_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        n_kv_groups: 2,
    };
    let output = cache
        .decode(TEST_LAYER, &k_dec, &v_dec, &q_dec, &config)
        .unwrap();

    if let DecodeOutput::Dequantized(result) = output {
        // Reconstruct original K by concatenating prefill + decode
        let k_orig = Tensor::cat(&[&k_pre, &k_dec], 2).unwrap();
        let sim = cosine_sim(&result.k, &k_orig);
        assert!(
            sim > 0.85,
            "PQO3 MaxNorm roundtrip cosine_sim={sim:.4}, expected > 0.85"
        );
    } else {
        panic!("Expected Dequantized on CPU");
    }
}

#[test]
fn pqo3_roundtrip_quality_l2norm() {
    let mut cache = PqoCache::new(pqo_config(BITS, QuantNormMode::L2Norm));

    let (k_pre, v_pre) = make_kv(4, 7.0);
    let q = make_q(4);
    cache.prefill(TEST_LAYER, &k_pre, &v_pre, &q).unwrap();

    let (k_dec, v_dec) = make_kv(1, 8.0);
    let q_dec = make_q(1);
    let config = AttendConfig {
        softmax_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        n_kv_groups: 2,
    };
    let output = cache
        .decode(TEST_LAYER, &k_dec, &v_dec, &q_dec, &config)
        .unwrap();

    if let DecodeOutput::Dequantized(result) = output {
        let k_orig = Tensor::cat(&[&k_pre, &k_dec], 2).unwrap();
        let sim = cosine_sim(&result.k, &k_orig);
        assert!(
            sim > 0.83,
            "PQO3 L2Norm roundtrip cosine_sim={sim:.4}, expected > 0.83"
        );
    } else {
        panic!("Expected Dequantized on CPU");
    }
}

#[test]
fn pqo4_roundtrip_quality_maxnorm() {
    let mut cache = PqoCache::new(pqo_config(4, QuantNormMode::MaxNorm));

    let (k_pre, v_pre) = make_kv(4, 9.0);
    let q = make_q(4);
    cache.prefill(TEST_LAYER, &k_pre, &v_pre, &q).unwrap();

    let (k_dec, v_dec) = make_kv(1, 10.0);
    let q_dec = make_q(1);
    let config = AttendConfig {
        softmax_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        n_kv_groups: 2,
    };
    let output = cache
        .decode(TEST_LAYER, &k_dec, &v_dec, &q_dec, &config)
        .unwrap();

    if let DecodeOutput::Dequantized(result) = output {
        let k_orig = Tensor::cat(&[&k_pre, &k_dec], 2).unwrap();
        let sim = cosine_sim(&result.k, &k_orig);
        assert!(
            sim > 0.92,
            "PQO4 MaxNorm roundtrip cosine_sim={sim:.4}, expected > 0.92"
        );
    } else {
        panic!("Expected Dequantized on CPU");
    }
}

// -----------------------------------------------------------------------
// Lifecycle: reset, layers, memory
// -----------------------------------------------------------------------

#[test]
fn pqo3_reset_clears_all_layers() {
    let mut cache = PqoCache::new(pqo_config(BITS, QuantNormMode::MaxNorm));
    let (k, v) = make_kv(4, 11.0);
    let q = make_q(4);

    cache.prefill(0, &k, &v, &q).unwrap();
    cache.prefill(1, &k, &v, &q).unwrap();
    assert_eq!(cache.seq_len(0), 4);
    assert_eq!(cache.seq_len(1), 4);

    cache.reset().unwrap();
    assert_eq!(cache.seq_len(0), 0);
    assert_eq!(cache.seq_len(1), 0);
    assert_eq!(cache.memory_usage(), 0);
}

#[test]
fn pqo3_layers_are_independent() {
    let mut cache = PqoCache::new(pqo_config(BITS, QuantNormMode::MaxNorm));
    let (k4, v4) = make_kv(4, 12.0);
    let (k8, v8) = make_kv(8, 13.0);
    let q4 = make_q(4);
    let q8 = make_q(8);

    cache.prefill(0, &k4, &v4, &q4).unwrap();
    cache.prefill(1, &k8, &v8, &q8).unwrap();

    assert_eq!(cache.seq_len(0), 4);
    assert_eq!(cache.seq_len(1), 8);
}

#[test]
fn pqo3_memory_usage_increases_with_tokens() {
    let mut cache = PqoCache::new(pqo_config(BITS, QuantNormMode::MaxNorm));
    assert_eq!(cache.memory_usage(), 0);

    let (k, v) = make_kv(16, 14.0);
    let q = make_q(16);
    cache.prefill(TEST_LAYER, &k, &v, &q).unwrap();

    let usage = cache.memory_usage();
    assert!(usage > 0, "Memory usage should be > 0 after prefill");

    // PQO3: 3-bit packed (48 bytes/token for K indices) + F16 scales
    // Rough estimate: 2 * heads * seq * (packed_dim + num_blocks * 2)
    let expected_min = NUM_KV_HEADS * 16 * 48; // just K indices
    assert!(
        usage > expected_min,
        "Memory usage {usage} too low, expected > {expected_min}"
    );
}

#[test]
fn pqo3_multi_step_decode() {
    let mut cache = PqoCache::new(pqo_config(BITS, QuantNormMode::MaxNorm));
    let config = AttendConfig {
        softmax_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        n_kv_groups: 2,
    };

    // Prefill 4 tokens
    let (k_pre, v_pre) = make_kv(4, 15.0);
    let q_pre = make_q(4);
    cache.prefill(TEST_LAYER, &k_pre, &v_pre, &q_pre).unwrap();

    // Decode 10 tokens one by one
    for step in 0..10 {
        let (k_dec, v_dec) = make_kv(1, 16.0 + step as f32);
        let q_dec = make_q(1);
        let output = cache
            .decode(TEST_LAYER, &k_dec, &v_dec, &q_dec, &config)
            .unwrap();

        match output {
            DecodeOutput::Dequantized(result) => {
                let expected_seq = 4 + step + 1;
                assert_eq!(
                    result.k.dims(),
                    &[1, NUM_KV_HEADS, expected_seq, HEAD_DIM],
                    "Step {step}: wrong K shape"
                );
            }
            DecodeOutput::Fused(_) => panic!("CPU should not use fused path"),
        }
    }
    assert_eq!(cache.seq_len(TEST_LAYER), 14);
}

// -----------------------------------------------------------------------
// GPU tests (CUDA fused attention path)
// -----------------------------------------------------------------------

#[cfg(feature = "cuda")]
mod gpu_tests {
    use super::*;

    fn cuda_device() -> Device {
        Device::cuda_if_available(0).expect("CUDA device required for GPU tests")
    }

    /// Generate test data on GPU.
    fn make_kv_gpu(seq_len: usize, seed: f32) -> (Tensor, Tensor) {
        let (k, v) = make_kv(seq_len, seed);
        let dev = cuda_device();
        (k.to_device(&dev).unwrap(), v.to_device(&dev).unwrap())
    }

    fn make_q_gpu(seq_len: usize) -> Tensor {
        let q = make_q(seq_len);
        q.to_device(&cuda_device()).unwrap()
    }

    #[test]
    fn pqo3_gpu_decode_returns_fused() {
        let mut cache = PqoCache::new(pqo_config(BITS, QuantNormMode::MaxNorm));

        // Prefill on GPU
        let (k_pre, v_pre) = make_kv_gpu(8, 20.0);
        let q_pre = make_q_gpu(8);
        cache.prefill(TEST_LAYER, &k_pre, &v_pre, &q_pre).unwrap();

        // Decode 1 token on GPU — should use fused kernel
        let (k_dec, v_dec) = make_kv_gpu(1, 21.0);
        let q_dec = make_q_gpu(1);
        let config = AttendConfig {
            softmax_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
            n_kv_groups: 2,
        };
        let output = cache
            .decode(TEST_LAYER, &k_dec, &v_dec, &q_dec, &config)
            .unwrap();

        match output {
            DecodeOutput::Fused(tensor) => {
                let num_attn_heads = NUM_KV_HEADS * 2;
                assert_eq!(
                    tensor.dims(),
                    &[1, num_attn_heads, 1, HEAD_DIM],
                    "Fused output shape wrong"
                );
                // Verify output is not all zeros (kernel actually ran)
                let sum: f32 = tensor
                    .to_dtype(DType::F32)
                    .unwrap()
                    .abs()
                    .unwrap()
                    .sum_all()
                    .unwrap()
                    .to_scalar()
                    .unwrap();
                assert!(sum > 0.01, "Fused output is all zeros — kernel did not run");
            }
            DecodeOutput::Dequantized(_) => {
                panic!("GPU decode should use Fused path, got Dequantized");
            }
        }
    }

    #[test]
    fn pqo3_gpu_multi_step_decode_fused() {
        let mut cache = PqoCache::new(pqo_config(BITS, QuantNormMode::MaxNorm));
        let config = AttendConfig {
            softmax_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
            n_kv_groups: 2,
        };

        // Prefill
        let (k_pre, v_pre) = make_kv_gpu(4, 22.0);
        let q_pre = make_q_gpu(4);
        cache.prefill(TEST_LAYER, &k_pre, &v_pre, &q_pre).unwrap();

        // Decode 5 tokens
        for step in 0..5 {
            let (k_dec, v_dec) = make_kv_gpu(1, 23.0 + step as f32);
            let q_dec = make_q_gpu(1);
            let output = cache
                .decode(TEST_LAYER, &k_dec, &v_dec, &q_dec, &config)
                .unwrap();
            match output {
                DecodeOutput::Fused(tensor) => {
                    assert_eq!(
                        tensor.dims()[2],
                        1,
                        "Step {step}: Fused output should be single token"
                    );
                }
                DecodeOutput::Dequantized(_) => {
                    panic!("Step {step}: GPU decode should use Fused path");
                }
            }
        }
        assert_eq!(cache.seq_len(TEST_LAYER), 9);
    }

    #[test]
    fn pqo3_gpu_fused_quality_reasonable() {
        let mut cache = PqoCache::new(pqo_config(BITS, QuantNormMode::MaxNorm));
        let config = AttendConfig {
            softmax_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
            n_kv_groups: 2,
        };

        // Prefill + decode on GPU
        let (k_pre, v_pre) = make_kv_gpu(16, 30.0);
        let q_pre = make_q_gpu(16);
        cache.prefill(TEST_LAYER, &k_pre, &v_pre, &q_pre).unwrap();

        let (k_dec, v_dec) = make_kv_gpu(1, 31.0);
        let q_dec = make_q_gpu(1);
        let output = cache
            .decode(TEST_LAYER, &k_dec, &v_dec, &q_dec, &config)
            .unwrap();

        // Also compute dequantized path on CPU for comparison
        let mut cache_cpu = PqoCache::new(pqo_config(BITS, QuantNormMode::MaxNorm));
        let (k_pre_cpu, v_pre_cpu) = make_kv(16, 30.0);
        let q_pre_cpu = make_q(16);
        cache_cpu
            .prefill(TEST_LAYER, &k_pre_cpu, &v_pre_cpu, &q_pre_cpu)
            .unwrap();
        let (k_dec_cpu, v_dec_cpu) = make_kv(1, 31.0);
        let q_dec_cpu = make_q(1);
        let cpu_output = cache_cpu
            .decode(TEST_LAYER, &k_dec_cpu, &v_dec_cpu, &q_dec_cpu, &config)
            .unwrap();

        if let (DecodeOutput::Fused(gpu_out), DecodeOutput::Dequantized(cpu_result)) =
            (output, cpu_output)
        {
            // The GPU fused output is attention output.
            // The CPU path returns dequantized KV (not attention output).
            // We can only verify that the GPU output is non-trivial.
            let gpu_sum: f32 = gpu_out
                .to_dtype(DType::F32)
                .unwrap()
                .abs()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar()
                .unwrap();
            assert!(
                gpu_sum > 0.1,
                "GPU fused attention output too small: {gpu_sum}"
            );

            // Verify CPU dequant quality
            let k_orig = Tensor::cat(&[&k_pre_cpu, &k_dec_cpu], 2).unwrap();
            let sim = cosine_sim(&cpu_result.k, &k_orig);
            assert!(sim > 0.85, "CPU dequant quality too low: {sim}");
        }
    }
}
