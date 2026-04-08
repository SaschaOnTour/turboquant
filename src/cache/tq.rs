//! TurboQuant (TQ) compressed KV-cache with QJL correction.
//!
//! TQ = (bits-1)-bit PolarQuant + 1-bit QJL (Paper Algorithm 2).
//! Uses standard codebook (outlier_blocks=0) plus QJL bias correction
//! to achieve unbiased inner-product estimates.

use candle_core::{DType, Device, Result, Tensor};
use mistralrs_kv_cache::{AttendConfig, CompressedKVCache, DecodeOutput, DequantResult};

use super::config::{QuantNormMode, DEFAULT_QJL_SEED, QUANT_BLOCK_SIZE};
use super::precomputed::GpuPrecomputed;
use super::quantize_tensor::{polar_dequantize, polar_quantize};
use super::storage::CompressedStorage;

const BITS_PER_BYTE: usize = 8;

/// TurboQuant cache: (bits-1)-bit PolarQuant + 1-bit QJL correction.
pub struct TqCache {
    bits: u8,
    head_dim: usize,
    num_kv_heads: usize,
    norm_mode: QuantNormMode,
    storage: CompressedStorage,
    precomputed: Option<GpuPrecomputed>,
    // QJL data per layer
    qjl_signs: Vec<Option<Tensor>>,
    qjl_norms: Vec<Option<Tensor>>,
}

impl TqCache {
    /// Create a new TQ cache.
    pub fn new(
        bits: u8,
        head_dim: usize,
        num_kv_heads: usize,
        num_layers: usize,
        norm_mode: QuantNormMode,
    ) -> Self {
        Self {
            bits,
            head_dim,
            num_kv_heads,
            norm_mode,
            storage: CompressedStorage::new(num_kv_heads, head_dim, bits, num_layers),
            precomputed: None,
            qjl_signs: vec![None; num_layers],
            qjl_norms: vec![None; num_layers],
        }
    }

    fn ensure_precomputed(&mut self, device: &Device) -> Result<()> {
        if self.precomputed.is_some() {
            return Ok(());
        }
        self.precomputed = Some(GpuPrecomputed::new(
            self.bits,
            self.head_dim,
            self.norm_mode,
            0,    // TQ uses standard codebook (outlier_blocks=0)
            true, // QJL enabled
            device,
        )?);
        Ok(())
    }

    /// Ensure QJL buffers have capacity for `needed` tokens.
    fn ensure_qjl_capacity(
        &mut self,
        layer: usize,
        needed: usize,
        device: &Device,
    ) -> Result<()> {
        let signs_per_head = self.head_dim / BITS_PER_BYTE;
        let heads = self.num_kv_heads;
        let current_cap = self.qjl_signs[layer]
            .as_ref()
            .map_or(0, |t| t.dims()[1]);

        if current_cap >= needed {
            return Ok(());
        }

        let grow = (needed / 4).max(128);
        let new_cap = needed + grow;
        let old_seq = self.storage.seq_len(layer);

        let new_signs = Tensor::zeros((heads, new_cap, signs_per_head), DType::U8, device)?;
        let new_norms = Tensor::zeros((heads, new_cap), DType::F16, device)?;

        if old_seq > 0 {
            if let Some(ref old) = self.qjl_signs[layer] {
                new_signs.slice_set(&old.narrow(1, 0, old_seq)?, 1, 0)?;
            }
            if let Some(ref old) = self.qjl_norms[layer] {
                new_norms.slice_set(&old.narrow(1, 0, old_seq)?, 1, 0)?;
            }
        }

        self.qjl_signs[layer] = Some(new_signs);
        self.qjl_norms[layer] = Some(new_norms);
        Ok(())
    }

    /// Quantize + store + compute QJL signs/norms for new tokens.
    fn quantize_and_store(
        &mut self,
        layer: usize,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<(usize, usize)> {
        let device = k.device().clone();
        self.ensure_precomputed(&device)?;

        let head_dim = self.head_dim;
        let num_kv_heads = self.num_kv_heads;
        let new_seq_len = k.dims()[2];
        let old_seq_len = self.storage.seq_len(layer);
        let total_seq_len = old_seq_len + new_seq_len;

        let k_flat = k.squeeze(0)?.to_dtype(DType::F32)?
            .reshape((num_kv_heads * new_seq_len, head_dim))?;
        let v_flat = v.squeeze(0)?.to_dtype(DType::F32)?
            .reshape((num_kv_heads * new_seq_len, head_dim))?;

        self.storage.ensure_capacity(layer, total_seq_len, &device)?;
        self.ensure_qjl_capacity(layer, total_seq_len, &device)?;

        let pre = self.precomputed.as_ref().unwrap();
        let packed_dim = self.storage.packed_dim();
        let num_blocks = self.storage.num_blocks();

        let (k_idx, k_sc) = polar_quantize(&k_flat, head_dim, self.bits, self.norm_mode, 0, pre)?;
        let (v_idx, v_sc) = polar_quantize(&v_flat, head_dim, self.bits, self.norm_mode, 0, pre)?;

        let k_idx_r = k_idx.reshape((num_kv_heads, new_seq_len, packed_dim))?;
        let v_idx_r = v_idx.reshape((num_kv_heads, new_seq_len, packed_dim))?;
        let k_sc_r = k_sc.reshape((num_kv_heads, new_seq_len, num_blocks))?;
        let v_sc_r = v_sc.reshape((num_kv_heads, new_seq_len, num_blocks))?;

        self.storage.append(layer, old_seq_len, &k_idx_r, &k_sc_r, &v_idx_r, &v_sc_r, new_seq_len)?;

        // Compute QJL signs + residual norms
        let n_vecs = num_kv_heads * new_seq_len;
        let k_idx_flat = k_idx.reshape((n_vecs, packed_dim))?;
        let k_sc_flat = k_sc.reshape((n_vecs, num_blocks))?;
        let k_dequant = polar_dequantize(&k_idx_flat, &k_sc_flat, head_dim, self.bits, 0, pre)?;

        let signs_per_head = head_dim / BITS_PER_BYTE;
        let (signs_tensor, norms_tensor) = compute_qjl_signs_and_norms(
            &k_flat, &k_dequant, n_vecs, head_dim, signs_per_head,
        )?;

        let signs_r = signs_tensor.reshape((num_kv_heads, new_seq_len, signs_per_head))?;
        let norms_r = norms_tensor.reshape((num_kv_heads, new_seq_len))?;

        self.qjl_signs[layer].as_ref().unwrap().slice_set(&signs_r, 1, old_seq_len)?;
        self.qjl_norms[layer].as_ref().unwrap().slice_set(&norms_r, 1, old_seq_len)?;

        Ok((old_seq_len, total_seq_len))
    }

    /// Compute QJL logit bias for attention correction.
    fn compute_logit_bias(
        &self,
        layer: usize,
        q: &Tensor,
    ) -> Result<Tensor> {
        let head_dim = self.head_dim;
        let total_seq = self.storage.seq_len(layer);
        let pre = self.precomputed.as_ref().unwrap();
        let signs_per_head = head_dim / BITS_PER_BYTE;

        // q shape: [1, num_attn_heads, q_len, head_dim]
        let q_dims = q.dims4()?;
        let num_attn_heads = q_dims.1;
        let q_len = q_dims.2;

        let rademacher = pre.qjl_rademacher.as_ref()
            .ok_or_else(|| super::cache_err("QJL Rademacher matrix not precomputed"))?;
        let rademacher_t = rademacher.t()?;

        let sqrt_pi_over_2 = std::f64::consts::FRAC_PI_2.sqrt() as f32;
        let inv_sqrt_dim = 1.0 / (head_dim as f32).sqrt();
        let scale_factor = sqrt_pi_over_2 * inv_sqrt_dim;

        // Per KV-head: compute correction [q_len, kv_len]
        let mut head_corrections = Vec::with_capacity(self.num_kv_heads);
        let n_kv_groups = num_attn_heads / self.num_kv_heads;

        for kv_head in 0..self.num_kv_heads {
            // Signs [kv_len, signs_per_head] → unpack to [kv_len, dim] as ±1.0
            let head_signs = self.qjl_signs[layer].as_ref().unwrap()
                .narrow(0, kv_head, 1)?.narrow(1, 0, total_seq)?.squeeze(0)?;
            let head_norms = self.qjl_norms[layer].as_ref().unwrap()
                .narrow(0, kv_head, 1)?.narrow(1, 0, total_seq)?.squeeze(0)?
                .to_dtype(DType::F32)?;

            // Unpack U8 signs to ±1.0 float
            let signs_u8 = head_signs.unsqueeze(2)?;
            let bit_masks = Tensor::from_vec(
                vec![1u8, 2, 4, 8, 16, 32, 64, 128], (1, 1, 8), q.device(),
            )?;
            let bits_set = signs_u8.to_dtype(DType::U32)?
                .broadcast_mul(&bit_masks.to_dtype(DType::U32)?)?;
            let bit_set = bits_set.ne(0u32)?.to_dtype(DType::F32)?;
            let signs_float = ((bit_set * 2.0)? - 1.0)?.reshape((total_seq, head_dim))?;
            let signs_float_t = signs_float.t()?; // [dim, kv_len]

            // Scale: c = norm * sqrt(pi/2) / sqrt(dim)
            let c = (head_norms * scale_factor as f64)?;
            let c_row = c.unsqueeze(0)?; // [1, kv_len]

            // For each query head in this KV group
            for qh in 0..n_kv_groups {
                let attn_head = kv_head * n_kv_groups + qh;
                let q_head = q.narrow(1, attn_head, 1)?.squeeze(0)?.squeeze(0)?
                    .to_dtype(DType::F32)?; // [q_len, dim]

                // r_q = q @ R^T  [q_len, dim]
                let r_q = q_head.matmul(&rademacher_t)?;
                // raw = r_q @ signs^T  [q_len, kv_len]
                let raw = r_q.matmul(&signs_float_t)?;
                // correction = raw * c  [q_len, kv_len]
                let corr = raw.broadcast_mul(&c_row)?;
                head_corrections.push(corr.unsqueeze(0)?); // [1, q_len, kv_len]
            }
        }

        // Stack: [num_attn_heads, q_len, kv_len]
        let refs: Vec<&Tensor> = head_corrections.iter().collect();
        let combined = Tensor::cat(&refs, 0)?;
        // → [1, num_attn_heads, q_len, kv_len], match query dtype
        combined.unsqueeze(0)?.to_dtype(q.dtype())
    }

    fn dequantize_full(&self, layer: usize, orig_dtype: DType) -> Result<(Tensor, Tensor)> {
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

        let full_k = polar_dequantize(&all_ki, &all_ks, head_dim, self.bits, 0, pre)?
            .reshape((1, num_kv_heads, total_seq, head_dim))?.to_dtype(orig_dtype)?;
        let full_v = polar_dequantize(&all_vi, &all_vs, head_dim, self.bits, 0, pre)?
            .reshape((1, num_kv_heads, total_seq, head_dim))?.to_dtype(orig_dtype)?;

        Ok((full_k, full_v))
    }
}

impl CompressedKVCache for TqCache {
    fn prefill(&mut self, layer: usize, k: &Tensor, v: &Tensor, q: &Tensor) -> Result<DequantResult> {
        let orig_dtype = k.dtype();
        let (old_seq_len, _total) = self.quantize_and_store(layer, k, v)?;

        let (full_k, full_v) = if old_seq_len == 0 {
            (k.clone(), v.clone())
        } else {
            self.dequantize_full(layer, orig_dtype)?
        };

        let logit_bias = self.compute_logit_bias(layer, q)?;
        Ok(DequantResult { k: full_k, v: full_v, logit_bias: Some(logit_bias) })
    }

    fn decode(
        &mut self, layer: usize, k: &Tensor, v: &Tensor, q: &Tensor, _config: &AttendConfig,
    ) -> Result<DecodeOutput> {
        let orig_dtype = k.dtype();
        self.quantize_and_store(layer, k, v)?;

        // TQ always uses dequant path (no fused kernel with inline QJL yet)
        let (full_k, full_v) = self.dequantize_full(layer, orig_dtype)?;
        let logit_bias = self.compute_logit_bias(layer, q)?;

        Ok(DecodeOutput::Dequantized(DequantResult {
            k: full_k,
            v: full_v,
            logit_bias: Some(logit_bias),
        }))
    }

    fn seq_len(&self, layer: usize) -> usize { self.storage.seq_len(layer) }
    fn reset(&mut self) -> Result<()> { self.storage.reset(); Ok(()) }
    fn memory_usage(&self) -> usize { self.storage.memory_usage() }
}

/// Compute QJL sign bits + residual norms on CPU, then move to target device.
fn compute_qjl_signs_and_norms(
    original: &Tensor,
    dequantized: &Tensor,
    n_vecs: usize,
    head_dim: usize,
    signs_per_head: usize,
) -> Result<(Tensor, Tensor)> {
    let device = original.device().clone();
    let residual = (original - dequantized)?;
    let norms = residual.sqr()?.sum_keepdim(1)?.sqrt()?.squeeze(1)?.to_dtype(DType::F16)?;

    // Signs computed on CPU (hash-based, not GPU-parallelizable)
    let residual_cpu = residual.to_device(&Device::Cpu)?;
    let mut all_signs = vec![0u8; n_vecs * signs_per_head];
    for vec_idx in 0..n_vecs {
        let row_data: Vec<f32> = residual_cpu.narrow(0, vec_idx, 1)?.squeeze(0)?.to_vec1()?;
        let signs = crate::compute_qjl_signs(&row_data, head_dim, DEFAULT_QJL_SEED);
        let start = vec_idx * signs_per_head;
        all_signs[start..start + signs_per_head].copy_from_slice(&signs);
    }

    let signs = Tensor::from_vec(all_signs, n_vecs * signs_per_head, &Device::Cpu)?
        .to_device(&device)?;
    Ok((signs, norms))
}
