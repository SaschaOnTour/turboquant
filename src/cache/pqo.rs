//! PolarQuant Outlier (PQO) compressed KV-cache implementation.
//!
//! All blocks use the outlier (higher-bit) codebook — the recommended mode
//! for production use. Implements [`CompressedKVCache`] from `mistralrs-kv-cache`.

use candle_core::{DType, Device, Result, Tensor};
use mistralrs_kv_cache::{AttendConfig, CompressedKVCache, DecodeOutput, DequantResult};

use super::config::{QuantNormMode, QUANT_BLOCK_SIZE};
use super::precomputed::GpuPrecomputed;
use super::quantize_tensor::{polar_dequantize, polar_quantize};
use super::storage::CompressedStorage;

/// PolarQuant Outlier (PQO) compressed KV-cache.
///
/// All quantization blocks use the outlier (higher-bit) codebook,
/// providing the best quality among the PolarQuant variants.
/// Recommended mode: PQO3 (3-bit, outlier_blocks=all).
pub struct PqoCache {
    bits: u8,
    head_dim: usize,
    num_kv_heads: usize,
    norm_mode: QuantNormMode,
    outlier_blocks: usize,
    storage: CompressedStorage,
    precomputed: Option<GpuPrecomputed>,
}

impl PqoCache {
    /// Create a new PQO cache (all blocks use outlier codebook).
    ///
    /// `bits` is the total bit-width (3 or 4). The polar codebook uses `bits - 1`.
    pub fn new(
        bits: u8,
        head_dim: usize,
        num_kv_heads: usize,
        num_layers: usize,
        norm_mode: QuantNormMode,
    ) -> Self {
        Self::with_outlier_blocks(bits, head_dim, num_kv_heads, num_layers, norm_mode, usize::MAX)
    }

    /// Create a cache with a specific number of outlier blocks.
    ///
    /// - `outlier_blocks = 0`: PQ (plain PolarQuant, standard codebook only)
    /// - `outlier_blocks = usize::MAX`: PQO (all blocks use outlier codebook)
    ///
    /// # Panics
    ///
    /// Panics if `head_dim` is not divisible by `QUANT_BLOCK_SIZE` (32).
    pub fn with_outlier_blocks(
        bits: u8,
        head_dim: usize,
        num_kv_heads: usize,
        num_layers: usize,
        norm_mode: QuantNormMode,
        outlier_blocks: usize,
    ) -> Self {
        assert!(
            head_dim % QUANT_BLOCK_SIZE == 0,
            "head_dim ({head_dim}) must be divisible by QUANT_BLOCK_SIZE ({QUANT_BLOCK_SIZE}). \
             Models with head_dim={head_dim} are not supported by TurboQuant compression."
        );
        Self {
            bits,
            head_dim,
            num_kv_heads,
            norm_mode,
            outlier_blocks,
            storage: CompressedStorage::new(num_kv_heads, head_dim, bits, num_layers),
            precomputed: None,
        }
    }

    /// Ensure precomputed tensors are initialized on the given device.
    fn ensure_precomputed(&mut self, device: &Device) -> Result<()> {
        if self.precomputed.is_some() {
            return Ok(());
        }
        self.precomputed = Some(GpuPrecomputed::new(
            self.bits,
            self.head_dim,
            self.norm_mode,
            self.outlier_blocks,
            false, // no QJL (TqCache will handle QJL)
            device,
        )?);
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

        let num_kv_heads = self.num_kv_heads;
        let head_dim = self.head_dim;
        let new_seq_len = k.dims()[2];
        let old_seq_len = self.storage.seq_len(layer);
        let total_seq_len = old_seq_len + new_seq_len;

        // Flatten: [1, heads, seq, dim] -> [heads * seq, dim]
        let k_flat = k
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .reshape((num_kv_heads * new_seq_len, head_dim))?;
        let v_flat = v
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .reshape((num_kv_heads * new_seq_len, head_dim))?;

        self.storage.ensure_capacity(layer, total_seq_len, &device)?;
        let pre = self.precomputed.as_ref().unwrap();

        let (k_idx, k_sc) = polar_quantize(&k_flat, head_dim, self.bits, self.norm_mode, self.outlier_blocks, pre)?;
        let (v_idx, v_sc) = polar_quantize(&v_flat, head_dim, self.bits, self.norm_mode, self.outlier_blocks, pre)?;

        let packed_dim = self.storage.packed_dim();
        let num_blocks = self.storage.num_blocks();
        let k_idx = k_idx.reshape((num_kv_heads, new_seq_len, packed_dim))?;
        let v_idx = v_idx.reshape((num_kv_heads, new_seq_len, packed_dim))?;
        let k_sc = k_sc.reshape((num_kv_heads, new_seq_len, num_blocks))?;
        let v_sc = v_sc.reshape((num_kv_heads, new_seq_len, num_blocks))?;

        self.storage.append(layer, old_seq_len, &k_idx, &k_sc, &v_idx, &v_sc, new_seq_len)?;

        Ok((old_seq_len, total_seq_len))
    }

    /// Dequantize the full compressed cache for a layer.
    fn dequantize_full(
        &self,
        layer: usize,
        orig_dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        let total_seq = self.storage.seq_len(layer);
        let head_dim = self.head_dim;
        let num_kv_heads = self.num_kv_heads;
        let packed_dim = self.storage.packed_dim();
        let num_blocks = self.storage.num_blocks();
        let pre = self.precomputed.as_ref().unwrap();

        let ki = self.storage.k_indices(layer).unwrap();
        let ks = self.storage.k_scales(layer).unwrap();
        let vi = self.storage.v_indices(layer).unwrap();
        let vs = self.storage.v_scales(layer).unwrap();

        let all_ki = ki.narrow(1, 0, total_seq)?.reshape((num_kv_heads * total_seq, packed_dim))?;
        let all_ks = ks.narrow(1, 0, total_seq)?.reshape((num_kv_heads * total_seq, num_blocks))?;
        let all_vi = vi.narrow(1, 0, total_seq)?.reshape((num_kv_heads * total_seq, packed_dim))?;
        let all_vs = vs.narrow(1, 0, total_seq)?.reshape((num_kv_heads * total_seq, num_blocks))?;

        let full_k = polar_dequantize(&all_ki, &all_ks, head_dim, self.bits, self.outlier_blocks, pre)?
            .reshape((1, num_kv_heads, total_seq, head_dim))?
            .to_dtype(orig_dtype)?;
        let full_v = polar_dequantize(&all_vi, &all_vs, head_dim, self.bits, self.outlier_blocks, pre)?
            .reshape((1, num_kv_heads, total_seq, head_dim))?
            .to_dtype(orig_dtype)?;

        Ok((full_k, full_v))
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
            // First prefill: return originals (no old cache to dequant)
            Ok(DequantResult {
                k: k.clone(),
                v: v.clone(),
                logit_bias: None,
            })
        } else {
            // Subsequent prefill: dequant full cache
            let (full_k, full_v) = self.dequantize_full(layer, orig_dtype)?;
            Ok(DequantResult {
                k: full_k,
                v: full_v,
                logit_bias: None,
            })
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

        // CUDA: use fused attention kernel (dequant in shared memory, no full-dequant tensor)
        #[cfg(feature = "cuda")]
        if device.is_cuda() && self.storage.is_active(layer) {
            let pre = self.precomputed.as_ref().unwrap();
            let ki = self.storage.k_indices(layer).unwrap();
            let ks = self.storage.k_scales(layer).unwrap();
            let vi = self.storage.v_indices(layer).unwrap();
            let vs = self.storage.v_scales(layer).unwrap();
            let total_seq = self.storage.seq_len(layer);
            let kv_capacity = self.storage.capacity(layer);

            // Extract sign pattern for the kernel
            let sqrt_bs = (QUANT_BLOCK_SIZE as f64).sqrt();
            let sign_pattern = (pre.rotation_fwd.narrow(0, 0, 1)? * sqrt_bs)?
                .squeeze(0)?
                .to_dtype(DType::F32)?
                .contiguous()?;

            let q_squeezed = q.squeeze(0)?.squeeze(1)?.to_dtype(DType::F32)?.contiguous()?;
            let num_attention_heads = q_squeezed.dims()[0];

            let output = super::cuda::attention::fused_attention(
                &q_squeezed,
                ki,
                ks,
                vi,
                vs,
                &pre.outlier_centroids,
                &sign_pattern,
                num_attention_heads,
                self.num_kv_heads,
                self.head_dim,
                total_seq,
                kv_capacity,
                self.storage.packed_dim(),
                self.storage.num_blocks(),
                self.bits as usize,
                config.softmax_scale,
                &device,
            )?;

            return Ok(DecodeOutput::Fused(
                output.reshape((1, num_attention_heads, 1, self.head_dim))?
                    .to_dtype(orig_dtype)?,
            ));
        }

        // CPU/Metal: full dequantize + return for SDPA
        let (full_k, full_v) = self.dequantize_full(layer, orig_dtype)?;
        Ok(DecodeOutput::Dequantized(DequantResult {
            k: full_k,
            v: full_v,
            logit_bias: None,
        }))
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
