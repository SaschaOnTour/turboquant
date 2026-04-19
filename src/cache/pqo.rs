//! PolarQuant Outlier (PQO) compressed KV-cache implementation.
//!
//! All blocks use the outlier (higher-bit) codebook — the recommended mode
//! for production use. Implements [`CompressedKVCache`] from `mistralrs-kv-cache`.

use candle_core::{DType, Device, Result, Tensor};
use mistralrs_kv_cache::{AttendConfig, CompressedKVCache, DecodeOutput, DequantResult};
use parking_lot::Mutex;

use super::cache_err;
use super::common::{
    dequant_result, dequantize_full_impl, flatten_kv, make_quant_config, quantize_kv_pair,
    validate_and_make_metadata,
};
use super::config::{CacheConfig, QUANT_BLOCK_SIZE};
use super::precomputed::GpuPrecomputed;
use super::storage::{LayerStorage, QuantizedKV, StorageMetadata};
use super::{ensure_gpu_precomputed, PrecomputedState};

/// PolarQuant Outlier (PQO) compressed KV-cache.
///
/// All quantization blocks use the outlier (higher-bit) codebook,
/// providing the best quality among the PolarQuant variants.
/// Recommended mode: PQO3 (3-bit, outlier_blocks=all).
pub struct PqoCache {
    config: CacheConfig,
    metadata: StorageMetadata,
    precomputed: PrecomputedState,
    layers: Vec<Mutex<LayerStorage>>,
}

impl PqoCache {
    /// Create a new PQO/PQ/TQ cache from configuration.
    ///
    /// Returns an error if `head_dim` is not divisible by `QUANT_BLOCK_SIZE` (32).
    pub fn new(config: CacheConfig) -> candle_core::Result<Self> {
        let metadata = validate_and_make_metadata(&config)?;
        let layers = (0..config.num_layers)
            .map(|_| Mutex::new(LayerStorage::default()))
            .collect();
        Ok(Self {
            config,
            metadata,
            precomputed: PrecomputedState::default(),
            layers,
        })
    }

    /// Quantize new K/V and append to the locked layer.
    /// Returns (old_seq_len, new_total_seq_len).
    fn quantize_and_store(
        &self,
        layer_slot: &mut LayerStorage,
        k: &Tensor,
        v: &Tensor,
        pre: &GpuPrecomputed,
    ) -> Result<(usize, usize)> {
        let device = k.device().clone();

        let new_seq_len = k.dims()[2];
        let old_seq_len = layer_slot.seq_len();
        layer_slot.ensure_capacity(old_seq_len + new_seq_len, &self.metadata, &device)?;

        let (k_flat, v_flat) = flatten_kv(k, v, self.config.num_kv_heads, self.config.head_dim)?;
        let qc = make_quant_config(pre, &self.config);
        let (k_idx, k_sc, v_idx, v_sc) =
            quantize_kv_pair(&k_flat, &v_flat, self.config.norm_mode, &qc)?;

        let heads = self.config.num_kv_heads;
        let packed_dim = self.metadata.packed_dim();
        let num_blocks = self.metadata.num_blocks();
        let k_idx = k_idx.reshape((heads, new_seq_len, packed_dim))?;
        let v_idx = v_idx.reshape((heads, new_seq_len, packed_dim))?;
        let k_sc = k_sc.reshape((heads, new_seq_len, num_blocks))?;
        let v_sc = v_sc.reshape((heads, new_seq_len, num_blocks))?;

        let kv = QuantizedKV {
            k_indices: &k_idx,
            k_scales: &k_sc,
            v_indices: &v_idx,
            v_scales: &v_sc,
        };
        layer_slot.append(old_seq_len, &kv, new_seq_len)?;

        Ok((old_seq_len, old_seq_len + new_seq_len))
    }

    /// CUDA fused-attention decode path. Caller holds the layer lock.
    // qual:allow(TQ-003) — CUDA-only, tested via mistral.rs integration tests
    #[cfg(feature = "cuda")]
    fn decode_cuda(
        &self,
        layer_slot: &LayerStorage,
        pre: &GpuPrecomputed,
        q: &Tensor,
        softmax_scale: f32,
        orig_dtype: DType,
        device: &Device,
    ) -> Result<DecodeOutput> {
        let bufs = layer_slot
            .buffers()
            .ok_or_else(|| cache_err("layer buffers not initialized"))?;

        let sqrt_bs = (QUANT_BLOCK_SIZE as f64).sqrt();
        let sign_pattern = (pre.rotation_fwd.narrow(0, 0, 1)? * sqrt_bs)?
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .contiguous()?;

        let q_squeezed = q
            .squeeze(0)?
            .squeeze(1)?
            .to_dtype(DType::F32)?
            .contiguous()?;
        let num_attention_heads = q_squeezed.dims()[0];

        let output = super::cuda::attention::fused_attention(
            &super::cuda::attention::FusedAttentionParams {
                q: &q_squeezed,
                k_indices: bufs.k_indices,
                k_scales: bufs.k_scales,
                v_indices: bufs.v_indices,
                v_scales: bufs.v_scales,
                codebook: &pre.outlier_centroids,
                sign_pattern: &sign_pattern,
                num_attention_heads,
                num_kv_heads: self.config.num_kv_heads,
                head_dim: self.config.head_dim,
                kv_len: layer_slot.seq_len(),
                kv_stride: layer_slot.capacity(),
                packed_dim: self.metadata.packed_dim(),
                num_qblocks: self.metadata.num_blocks(),
                bits: self.config.bits as usize,
                softmax_scale,
                device,
            },
        )?;

        Ok(DecodeOutput::Fused(
            output
                .reshape((1, num_attention_heads, 1, self.config.head_dim))?
                .to_dtype(orig_dtype)?,
        ))
    }

    // qual:allow(TQ-003) — tested via cache_pqo_tests integration tests
    fn dequantize_full(
        &self,
        layer_slot: &LayerStorage,
        pre: &GpuPrecomputed,
        orig_dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        let qc = make_quant_config(pre, &self.config);
        dequantize_full_impl(layer_slot, &self.metadata, &qc, orig_dtype)
    }

    /// Borrow-check `layer` and return the per-layer mutex. Returns a
    /// `candle_core::Error` instead of panicking when `layer >= num_layers`.
    fn layer_mutex(&self, layer: usize) -> Result<&Mutex<LayerStorage>> {
        self.layers.get(layer).ok_or_else(|| {
            super::cache_err(format!(
                "layer index {layer} out of range (cache has {} layers)",
                self.layers.len()
            ))
        })
    }
}

impl CompressedKVCache for PqoCache {
    fn prefill(&self, layer: usize, k: &Tensor, v: &Tensor, _q: &Tensor) -> Result<DequantResult> {
        let orig_dtype = k.dtype();
        let pre = ensure_gpu_precomputed(&self.precomputed, &self.config, k.device())?;
        let mut guard = self.layer_mutex(layer)?.lock();
        let (old_seq_len, _total) = self.quantize_and_store(&mut guard, k, v, pre)?;

        if old_seq_len == 0 {
            Ok(dequant_result(k.clone(), v.clone()))
        } else {
            let (full_k, full_v) = self.dequantize_full(&guard, pre, orig_dtype)?;
            Ok(dequant_result(full_k, full_v))
        }
    }

    fn decode(
        &self,
        layer: usize,
        k: &Tensor,
        v: &Tensor,
        q: &Tensor,
        _config: &AttendConfig,
    ) -> Result<DecodeOutput> {
        let device = k.device().clone();
        let orig_dtype = k.dtype();
        let pre = ensure_gpu_precomputed(&self.precomputed, &self.config, &device)?;
        let mut guard = self.layer_mutex(layer)?.lock();
        self.quantize_and_store(&mut guard, k, v, pre)?;

        #[cfg(feature = "cuda")]
        if device.is_cuda() && guard.is_active() {
            return self.decode_cuda(&guard, pre, q, _config.softmax_scale, orig_dtype, &device);
        }

        // CPU/Metal: full dequantize + return for SDPA
        let _ = q;
        let (full_k, full_v) = self.dequantize_full(&guard, pre, orig_dtype)?;
        Ok(DecodeOutput::Dequantized(dequant_result(full_k, full_v)))
    }

    /// Returns 0 for out-of-range `layer` rather than panicking — the trait
    /// signature is infallible so callers cannot distinguish "not yet
    /// populated" from "invalid index" anyway.
    fn seq_len(&self, layer: usize) -> usize {
        self.layers
            .get(layer)
            .map(|m| m.lock().seq_len())
            .unwrap_or(0)
    }

    fn reset(&self) -> Result<()> {
        for m in &self.layers {
            m.lock().reset();
        }
        Ok(())
    }

    fn memory_usage(&self) -> usize {
        self.layers
            .iter()
            .map(|m| m.lock().memory_usage(&self.metadata))
            .sum()
    }
}
