//! PqoCache GPU decode tests — fused CUDA kernel path.
//!
//! Extracted from the former `cache_pqo_gpu_tests.rs`.

#![cfg(all(feature = "candle", feature = "cuda"))]

use candle_core::{DType, Device, Tensor};
use mistralrs_kv_cache::{AttendConfig, CompressedKVCache, DecodeOutput};
use turboquant::cache::config::QuantNormMode;
use turboquant::cache::{CacheConfig, PqoCache};
use turboquant::test_utils::{make_kv as shared_make_kv, make_q as shared_make_q};

const HEAD_DIM: usize = 128;
const NUM_KV_HEADS: usize = 8;
const NUM_ATTN_HEADS: usize = NUM_KV_HEADS * 2;
const NUM_LAYERS: usize = 2;
const BITS: u8 = 3;
const TEST_LAYER: usize = 0;
const N_KV_GROUPS: usize = 2;

fn pqo_config() -> CacheConfig {
    CacheConfig {
        bits: BITS,
        head_dim: HEAD_DIM,
        num_kv_heads: NUM_KV_HEADS,
        num_layers: NUM_LAYERS,
        norm_mode: QuantNormMode::MaxNorm,
        outlier_blocks: usize::MAX,
    }
}

fn cuda_device() -> Device {
    Device::cuda_if_available(0).expect("CUDA device required for GPU tests")
}

fn make_kv_gpu(seq_len: usize, seed: u32) -> (Tensor, Tensor) {
    let (k, v) = shared_make_kv(seq_len, NUM_KV_HEADS, HEAD_DIM, seed);
    let dev = cuda_device();
    (k.to_device(&dev).unwrap(), v.to_device(&dev).unwrap())
}

fn make_q_gpu(seq_len: usize) -> Tensor {
    let q = shared_make_q(seq_len, NUM_ATTN_HEADS, HEAD_DIM);
    q.to_device(&cuda_device()).unwrap()
}

#[test]
fn pqo3_gpu_decode_returns_fused() -> candle_core::Result<()> {
    let cache = PqoCache::new(pqo_config())?;

    let (k_pre, v_pre) = make_kv_gpu(8, 20);
    let q_pre = make_q_gpu(8);
    cache.prefill(TEST_LAYER, &k_pre, &v_pre, &q_pre).unwrap();

    let (k_dec, v_dec) = make_kv_gpu(1, 21);
    let q_dec = make_q_gpu(1);
    let config = AttendConfig {
        softmax_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        n_kv_groups: N_KV_GROUPS,
    };
    let output = cache
        .decode(TEST_LAYER, &k_dec, &v_dec, &q_dec, &config)
        .unwrap();

    match output {
        DecodeOutput::Fused(tensor) => {
            assert_eq!(
                tensor.dims(),
                &[1, NUM_ATTN_HEADS, 1, HEAD_DIM],
                "Fused output shape wrong"
            );
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
    Ok(())
}

#[test]
fn pqo3_gpu_multi_step_decode_fused() -> candle_core::Result<()> {
    let cache = PqoCache::new(pqo_config())?;
    let config = AttendConfig {
        softmax_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        n_kv_groups: N_KV_GROUPS,
    };

    let (k_pre, v_pre) = make_kv_gpu(4, 22);
    let q_pre = make_q_gpu(4);
    cache.prefill(TEST_LAYER, &k_pre, &v_pre, &q_pre).unwrap();

    for step in 0..5 {
        let (k_dec, v_dec) = make_kv_gpu(1, 23 + step as u32);
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
    Ok(())
}
