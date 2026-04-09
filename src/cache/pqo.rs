//! PolarQuant Outlier (PQO) compressed KV-cache implementation.
//!
//! All blocks use the outlier (higher-bit) codebook — the recommended mode
//! for production use. Implements [`CompressedKVCache`] from `mistralrs-kv-cache`.

use candle_core::{DType, Device, Result, Tensor};
use mistralrs_kv_cache::{AttendConfig, CompressedKVCache, DecodeOutput, DequantResult};

use super::common::{
    dequant_result, dequantize_full_impl, flatten_kv, make_quant_config, quantize_kv_pair,
};
use super::config::{CacheConfig, QUANT_BLOCK_SIZE};
use super::precomputed::GpuPrecomputed;
use super::storage::{CompressedStorage, QuantizedKV};

/// PolarQuant Outlier (PQO) compressed KV-cache.
///
/// All quantization blocks use the outlier (higher-bit) codebook,
/// providing the best quality among the PolarQuant variants.
/// Recommended mode: PQO3 (3-bit, outlier_blocks=all).
pub struct PqoCache {
    config: CacheConfig,
    storage: CompressedStorage,
    precomputed: Option<GpuPrecomputed>,
}

impl PqoCache {
    /// Create a new PQO/PQ/TQ cache from configuration.
    ///
    /// # Panics
    ///
    /// Panics if `head_dim` is not divisible by `QUANT_BLOCK_SIZE` (32).
    pub fn new(config: CacheConfig) -> Self {
        assert!(
            config.head_dim % QUANT_BLOCK_SIZE == 0,
            "head_dim ({}) must be divisible by QUANT_BLOCK_SIZE ({QUANT_BLOCK_SIZE}). \
             Models with head_dim={} are not supported by TurboQuant compression.",
            config.head_dim,
            config.head_dim
        );
        let storage = CompressedStorage::new(
            config.num_kv_heads,
            config.head_dim,
            config.bits,
            config.num_layers,
        );
        Self {
            config,
            storage,
            precomputed: None,
        }
    }

    /// Ensure precomputed tensors are initialized on the given device.
    fn ensure_precomputed(&mut self, device: &Device) -> Result<()> {
        if self.precomputed.is_some() {
            return Ok(());
        }
        self.precomputed = Some(GpuPrecomputed::new(&self.config, device)?);
        Ok(())
    }

    /// Quantize new K/V and store in compressed buffers.
    /// Returns the old sequence length (offset for append).
    fn quantize_and_store(
        &mut self,
        layer: usize,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<(usize, usize)> {
        let device = k.device().clone();
        self.ensure_precomputed(&device)?;

        let new_seq_len = k.dims()[2];
        let old_seq_len = self.storage.seq_len(layer);
        self.storage
            .ensure_capacity(layer, old_seq_len + new_seq_len, &device)?;

        let (k_flat, v_flat) = flatten_kv(k, v, self.config.num_kv_heads, self.config.head_dim)?;
        let qc = make_quant_config(&self.precomputed, &self.config)?;
        let (k_idx, k_sc, v_idx, v_sc) =
            quantize_kv_pair(&k_flat, &v_flat, self.config.norm_mode, &qc)?;

        let heads = self.config.num_kv_heads;
        let packed_dim = self.storage.packed_dim();
        let num_blocks = self.storage.num_blocks();
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
        self.storage.append(layer, old_seq_len, &kv, new_seq_len)?;

        Ok((old_seq_len, old_seq_len + new_seq_len))
    }

    /// Dequantize the full compressed cache for a layer.
    /// CUDA fused-attention decode path.
    // qual:allow(TQ-003) — CUDA-only, tested via mistral.rs integration tests
    #[cfg(feature = "cuda")]
    fn decode_cuda(
        &self,
        layer: usize,
        q: &Tensor,
        softmax_scale: f32,
        orig_dtype: DType,
        device: &Device,
    ) -> Result<DecodeOutput> {
        use super::cache_err;
        let qc = make_quant_config(&self.precomputed, &self.config)?;
        let pre = qc.pre;
        let ki = self
            .storage
            .k_indices(layer)
            .ok_or_else(|| cache_err("k_indices not initialized"))?;
        let ks = self
            .storage
            .k_scales(layer)
            .ok_or_else(|| cache_err("k_scales not initialized"))?;
        let vi = self
            .storage
            .v_indices(layer)
            .ok_or_else(|| cache_err("v_indices not initialized"))?;
        let vs = self
            .storage
            .v_scales(layer)
            .ok_or_else(|| cache_err("v_scales not initialized"))?;

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
                k_indices: ki,
                k_scales: ks,
                v_indices: vi,
                v_scales: vs,
                codebook: &pre.outlier_centroids,
                sign_pattern: &sign_pattern,
                num_attention_heads,
                num_kv_heads: self.config.num_kv_heads,
                head_dim: self.config.head_dim,
                kv_len: self.storage.seq_len(layer),
                kv_stride: self.storage.capacity(layer),
                packed_dim: self.storage.packed_dim(),
                num_qblocks: self.storage.num_blocks(),
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
    fn dequantize_full(&self, layer: usize, orig_dtype: DType) -> Result<(Tensor, Tensor)> {
        let qc = make_quant_config(&self.precomputed, &self.config)?;
        dequantize_full_impl(&self.storage, &qc, layer, orig_dtype)
    }
}

impl CompressedKVCache for PqoCache {
    fn prefill(
        &mut self,
        layer: usize,
        k: &Tensor,
        v: &Tensor,
        _q: &Tensor,
    ) -> Result<DequantResult> {
        let orig_dtype = k.dtype();
        let (old_seq_len, _total) = self.quantize_and_store(layer, k, v)?;

        if old_seq_len == 0 {
            Ok(dequant_result(k.clone(), v.clone()))
        } else {
            let (full_k, full_v) = self.dequantize_full(layer, orig_dtype)?;
            Ok(dequant_result(full_k, full_v))
        }
    }

    fn decode(
        &mut self,
        layer: usize,
        k: &Tensor,
        v: &Tensor,
        q: &Tensor,
        config: &AttendConfig,
    ) -> Result<DecodeOutput> {
        let device = k.device().clone();
        let orig_dtype = k.dtype();
        self.quantize_and_store(layer, k, v)?;

        #[cfg(feature = "cuda")]
        if device.is_cuda() && self.storage.is_active(layer) {
            return self.decode_cuda(layer, q, config.softmax_scale, orig_dtype, &device);
        }

        // CPU/Metal: full dequantize + return for SDPA
        let (full_k, full_v) = self.dequantize_full(layer, orig_dtype)?;
        Ok(DecodeOutput::Dequantized(dequant_result(full_k, full_v)))
    }

    fn seq_len(&self, layer: usize) -> usize {
        self.storage.seq_len(layer)
    }

    fn reset(&mut self) -> Result<()> {
        self.storage.reset();
        Ok(())
    }

    fn memory_usage(&self) -> usize {
        self.storage.memory_usage()
    }
}
