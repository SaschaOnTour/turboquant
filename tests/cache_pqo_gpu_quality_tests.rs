//! PqoCache GPU quality test — verify the fused CUDA path produces
//! non-trivial attention output. CPU dequant quality is covered separately
//! in `cache_pqo_roundtrip_tests`.
//!
//! Extracted from the former `cache_pqo_gpu_tests.rs`.

#![cfg(all(feature = "candle", feature = "cuda"))]

use candle_core::{DType, Device};
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
const PREFILL_LEN: usize = 16;
const MIN_GPU_OUTPUT_ABS_SUM: f32 = 0.1;

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

#[test]
fn pqo3_gpu_fused_output_is_nontrivial() -> candle_core::Result<()> {
    let dev = Device::cuda_if_available(0).expect("CUDA device required");
    let (k_cpu, v_cpu) = shared_make_kv(PREFILL_LEN, NUM_KV_HEADS, HEAD_DIM, 30);
    let q_cpu = shared_make_q(PREFILL_LEN, NUM_ATTN_HEADS, HEAD_DIM);
    let (k_pre, v_pre) = (k_cpu.to_device(&dev)?, v_cpu.to_device(&dev)?);
    let q_pre = q_cpu.to_device(&dev)?;

    let (k_dec_cpu, v_dec_cpu) = shared_make_kv(1, NUM_KV_HEADS, HEAD_DIM, 31);
    let q_dec_cpu = shared_make_q(1, NUM_ATTN_HEADS, HEAD_DIM);
    let k_dec = k_dec_cpu.to_device(&dev)?;
    let v_dec = v_dec_cpu.to_device(&dev)?;
    let q_dec = q_dec_cpu.to_device(&dev)?;

    let cache = PqoCache::new(pqo_config())?;
    cache.prefill(TEST_LAYER, &k_pre, &v_pre, &q_pre).unwrap();

    let config = AttendConfig {
        softmax_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        n_kv_groups: N_KV_GROUPS,
    };
    let output = cache
        .decode(TEST_LAYER, &k_dec, &v_dec, &q_dec, &config)
        .unwrap();

    let DecodeOutput::Fused(gpu_out) = output else {
        panic!("GPU decode should use Fused path, got Dequantized");
    };

    let abs_sum: f32 = gpu_out
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_scalar()?;
    assert!(
        abs_sum > MIN_GPU_OUTPUT_ABS_SUM,
        "GPU fused attention output too small: abs_sum={abs_sum}"
    );
    Ok(())
}
