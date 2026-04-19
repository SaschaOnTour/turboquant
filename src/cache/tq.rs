//! TurboQuant (TQ) compressed KV-cache with QJL correction.
//!
//! TQ = (bits-1)-bit PolarQuant + 1-bit QJL (Paper Algorithm 2).
//! Uses standard codebook (outlier_blocks=0) plus QJL bias correction
//! to achieve unbiased inner-product estimates.

use candle_core::{DType, Device, Result, Tensor};
use mistralrs_kv_cache::{AttendConfig, CompressedKVCache, DecodeOutput, DequantResult};
use parking_lot::Mutex;

use super::cache_err;
use super::common::{
    dequantize_full_impl, flatten_kv, make_quant_config, quantize_kv_pair,
    validate_and_make_metadata,
};
use super::config::{CacheConfig, BITS_PER_BYTE, DEFAULT_QJL_SEED};
use super::precomputed::GpuPrecomputed;
use super::quantize_tensor::polar_dequantize;
use super::storage::{LayerStorage, QuantizedKV, StorageMetadata};
use super::{ensure_gpu_precomputed, PrecomputedState};

/// Minimum growth increment when expanding QJL sign/norm buffers.
const MIN_QJL_GROW: usize = 128;

/// Per-layer state for TqCache: quantized storage + QJL auxiliary data.
#[derive(Default)]
struct TqLayer {
    storage: LayerStorage,
    qjl_signs: Option<Tensor>,
    qjl_norms: Option<Tensor>,
}

/// TurboQuant cache: (bits-1)-bit PolarQuant + 1-bit QJL correction.
pub struct TqCache {
    config: CacheConfig,
    metadata: StorageMetadata,
    precomputed: PrecomputedState,
    layers: Vec<Mutex<TqLayer>>,
}

impl TqCache {
    /// Create a new TQ cache.
    ///
    /// Returns an error if `head_dim` is not divisible by `QUANT_BLOCK_SIZE` (32).
    pub fn new(config: CacheConfig) -> candle_core::Result<Self> {
        let metadata = validate_and_make_metadata(&config)?;
        let layers = (0..config.num_layers)
            .map(|_| Mutex::new(TqLayer::default()))
            .collect();
        Ok(Self {
            config,
            metadata,
            precomputed: PrecomputedState::default(),
            layers,
        })
    }

    /// Ensure QJL buffers for the locked layer have capacity for `needed` tokens.
    fn ensure_qjl_capacity(
        &self,
        layer_slot: &mut TqLayer,
        needed: usize,
        device: &Device,
    ) -> Result<()> {
        let signs_per_head = self.config.head_dim / BITS_PER_BYTE;
        let heads = self.config.num_kv_heads;
        let current_cap = layer_slot.qjl_signs.as_ref().map_or(0, |t| t.dims()[1]);

        if current_cap >= needed {
            return Ok(());
        }

        let grow = (needed / 4).max(MIN_QJL_GROW);
        let new_cap = needed + grow;
        let old_seq = layer_slot.storage.seq_len();

        let new_signs = Tensor::zeros((heads, new_cap, signs_per_head), DType::U8, device)?;
        let new_norms = Tensor::zeros((heads, new_cap), DType::F16, device)?;

        if old_seq > 0 {
            if let Some(ref old) = layer_slot.qjl_signs {
                new_signs.slice_set(&old.narrow(1, 0, old_seq)?, 1, 0)?;
            }
            if let Some(ref old) = layer_slot.qjl_norms {
                new_norms.slice_set(&old.narrow(1, 0, old_seq)?, 1, 0)?;
            }
        }

        layer_slot.qjl_signs = Some(new_signs);
        layer_slot.qjl_norms = Some(new_norms);
        Ok(())
    }

    /// Quantize + store + compute QJL signs/norms for new tokens. Caller holds the lock.
    // qual:allow(iosp) — orchestrator coordinating six steps: ensure capacity, flatten, quantize, reshape, append, compute QJL; splitting introduces param-passing overhead.
    fn quantize_and_store(
        &self,
        layer_slot: &mut TqLayer,
        k: &Tensor,
        v: &Tensor,
        pre: &GpuPrecomputed,
    ) -> Result<(usize, usize)> {
        let device = k.device().clone();

        let new_seq_len = k.dims()[2];
        let old_seq_len = layer_slot.storage.seq_len();
        let total_seq_len = old_seq_len + new_seq_len;
        layer_slot
            .storage
            .ensure_capacity(total_seq_len, &self.metadata, &device)?;
        self.ensure_qjl_capacity(layer_slot, total_seq_len, &device)?;

        let (k_flat, v_flat) = flatten_kv(k, v, self.config.num_kv_heads, self.config.head_dim)?;

        let qc = make_quant_config(pre, &self.config);
        let packed_dim = qc.packed_dim();
        let num_blocks = qc.num_blocks();

        let (k_idx, k_sc, v_idx, v_sc) =
            quantize_kv_pair(&k_flat, &v_flat, self.config.norm_mode, &qc)?;

        let heads = self.config.num_kv_heads;
        let k_idx_r = k_idx.reshape((heads, new_seq_len, packed_dim))?;
        let v_idx_r = v_idx.reshape((heads, new_seq_len, packed_dim))?;
        let k_sc_r = k_sc.reshape((heads, new_seq_len, num_blocks))?;
        let v_sc_r = v_sc.reshape((heads, new_seq_len, num_blocks))?;

        let kv = QuantizedKV {
            k_indices: &k_idx_r,
            k_scales: &k_sc_r,
            v_indices: &v_idx_r,
            v_scales: &v_sc_r,
        };
        layer_slot.storage.append(old_seq_len, &kv, new_seq_len)?;

        self.compute_and_store_qjl(layer_slot, &k_flat, &k_idx, &k_sc, &qc)?;

        Ok((old_seq_len, total_seq_len))
    }

    /// Compute QJL sign bits and residual norms, then store in the locked layer's QJL buffers.
    fn compute_and_store_qjl(
        &self,
        layer_slot: &mut TqLayer,
        k_flat: &Tensor,
        k_idx: &Tensor,
        k_sc: &Tensor,
        qc: &super::quantize_tensor::QuantConfig<'_>,
    ) -> Result<()> {
        let head_dim = self.config.head_dim;
        let num_kv_heads = self.config.num_kv_heads;
        let packed_dim = qc.packed_dim();
        let num_blocks = qc.num_blocks();
        let n_vecs = k_flat.dims()[0];
        let new_seq_len = n_vecs / num_kv_heads;
        let old_seq_len = layer_slot.storage.seq_len() - new_seq_len;

        let k_idx_flat = k_idx.reshape((n_vecs, packed_dim))?;
        let k_sc_flat = k_sc.reshape((n_vecs, num_blocks))?;
        let k_dequant = polar_dequantize(&k_idx_flat, &k_sc_flat, qc)?;

        let signs_per_head = head_dim / BITS_PER_BYTE;
        let (signs_tensor, norms_tensor) =
            compute_qjl_signs_and_norms(k_flat, &k_dequant, n_vecs, head_dim, signs_per_head)?;

        let signs_r = signs_tensor.reshape((num_kv_heads, new_seq_len, signs_per_head))?;
        let norms_r = norms_tensor.reshape((num_kv_heads, new_seq_len))?;

        layer_slot
            .qjl_signs
            .as_ref()
            .ok_or_else(|| cache_err("qjl_signs not initialized"))?
            .slice_set(&signs_r, 1, old_seq_len)?;
        layer_slot
            .qjl_norms
            .as_ref()
            .ok_or_else(|| cache_err("qjl_norms not initialized"))?
            .slice_set(&norms_r, 1, old_seq_len)?;

        Ok(())
    }

    /// Compute QJL logit bias for attention correction. Caller holds the lock.
    // qual:allow(TQ-003) — tested via cache_type_correctness integration tests
    fn compute_logit_bias(
        &self,
        layer_slot: &TqLayer,
        pre: &GpuPrecomputed,
        q: &Tensor,
    ) -> Result<Tensor> {
        let head_dim = self.config.head_dim;
        let total_seq = layer_slot.storage.seq_len();

        // q shape: [1, num_attn_heads, q_len, head_dim]
        let q_dims = q.dims4()?;
        let num_attn_heads = q_dims.1;

        let rademacher = pre
            .qjl_rademacher
            .as_ref()
            .ok_or_else(|| cache_err("QJL Rademacher matrix not precomputed"))?;
        let rademacher_t = rademacher.t()?;

        // Per KV-head: compute correction [q_len, kv_len]
        let mut head_corrections = Vec::with_capacity(self.config.num_kv_heads);
        let n_kv_groups = num_attn_heads / self.config.num_kv_heads;

        let qjl_signs = layer_slot
            .qjl_signs
            .as_ref()
            .ok_or_else(|| cache_err("qjl_signs not initialized"))?;
        let qjl_norms = layer_slot
            .qjl_norms
            .as_ref()
            .ok_or_else(|| cache_err("qjl_norms not initialized"))?;

        // Hoist per-head constants: bit masks tensor + scale factor
        let bit_masks =
            Tensor::from_vec(BYTE_BIT_MASKS.to_vec(), (1, 1, BITS_PER_BYTE), q.device())?;
        let sqrt_pi_over_2 = std::f64::consts::FRAC_PI_2.sqrt() as f32;
        let scale_factor = sqrt_pi_over_2 / (head_dim as f32).sqrt();

        for kv_head in 0..self.config.num_kv_heads {
            let (signs_float_t, c_row) = unpack_qjl_signs(
                qjl_signs,
                qjl_norms,
                kv_head,
                total_seq,
                head_dim,
                &bit_masks,
                scale_factor,
            )?;

            // For each query head in this KV group
            for qh in 0..n_kv_groups {
                let attn_head = kv_head * n_kv_groups + qh;
                let q_head = q
                    .narrow(1, attn_head, 1)?
                    .squeeze(0)?
                    .squeeze(0)?
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

    // qual:allow(TQ-003) — wrapper delegates to dequantize_full_impl, tested via integration tests
    fn dequantize_full(
        &self,
        layer_slot: &TqLayer,
        pre: &GpuPrecomputed,
        orig_dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        let qc = make_quant_config(pre, &self.config);
        dequantize_full_impl(&layer_slot.storage, &self.metadata, &qc, orig_dtype)
    }

    /// Borrow-check `layer` and return the per-layer mutex. Returns a
    /// `candle_core::Error` instead of panicking when `layer >= num_layers`.
    fn layer_mutex(&self, layer: usize) -> Result<&Mutex<TqLayer>> {
        self.layers.get(layer).ok_or_else(|| {
            cache_err(format!(
                "layer index {layer} out of range (cache has {} layers)",
                self.layers.len()
            ))
        })
    }
}

impl CompressedKVCache for TqCache {
    // qual:allow(iosp) — trait entry point orchestrating precomputed init, lock acquisition, quantize-and-store, dequantize, and logit-bias computation.
    fn prefill(&self, layer: usize, k: &Tensor, v: &Tensor, q: &Tensor) -> Result<DequantResult> {
        let orig_dtype = k.dtype();
        let pre = ensure_gpu_precomputed(&self.precomputed, &self.config, k.device())?;
        let mut guard = self.layer_mutex(layer)?.lock();
        let (old_seq_len, _total) = self.quantize_and_store(&mut guard, k, v, pre)?;

        let (full_k, full_v) = if old_seq_len == 0 {
            (k.clone(), v.clone())
        } else {
            self.dequantize_full(&guard, pre, orig_dtype)?
        };

        let logit_bias = self.compute_logit_bias(&guard, pre, q)?;
        Ok(DequantResult {
            k: full_k,
            v: full_v,
            logit_bias: Some(logit_bias),
        })
    }

    fn decode(
        &self,
        layer: usize,
        k: &Tensor,
        v: &Tensor,
        q: &Tensor,
        _config: &AttendConfig,
    ) -> Result<DecodeOutput> {
        let orig_dtype = k.dtype();
        let pre = ensure_gpu_precomputed(&self.precomputed, &self.config, k.device())?;
        let mut guard = self.layer_mutex(layer)?.lock();
        self.quantize_and_store(&mut guard, k, v, pre)?;

        // TQ always uses dequant path (no fused kernel with inline QJL yet)
        let (full_k, full_v) = self.dequantize_full(&guard, pre, orig_dtype)?;
        let logit_bias = self.compute_logit_bias(&guard, pre, q)?;

        Ok(DecodeOutput::Dequantized(DequantResult {
            k: full_k,
            v: full_v,
            logit_bias: Some(logit_bias),
        }))
    }

    /// Returns 0 for out-of-range `layer` rather than panicking — the trait
    /// signature is infallible so callers cannot distinguish "not yet
    /// populated" from "invalid index" anyway.
    fn seq_len(&self, layer: usize) -> usize {
        self.layers
            .get(layer)
            .map(|m| m.lock().storage.seq_len())
            .unwrap_or(0)
    }

    fn reset(&self) -> Result<()> {
        self.layers
            .iter()
            .for_each(|m| *m.lock() = TqLayer::default());
        Ok(())
    }

    fn memory_usage(&self) -> usize {
        self.layers
            .iter()
            .map(|m| {
                let g = m.lock();
                let storage_bytes = g.storage.memory_usage(&self.metadata);
                let qjl_bytes: usize = [g.qjl_signs.as_ref(), g.qjl_norms.as_ref()]
                    .iter()
                    .flatten()
                    .map(|t| t.elem_count() * t.dtype().size_in_bytes())
                    .sum();
                storage_bytes + qjl_bytes
            })
            .sum()
    }
}

/// Byte-level bit masks for unpacking U8 → individual bits.
const BYTE_BIT_MASKS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

/// Unpack QJL sign bits for one KV head and compute scaled correction weights.
///
/// Returns `(signs_float_t, c_row)`:
/// - `signs_float_t`: transposed ±1.0 sign matrix `[head_dim, kv_len]`
/// - `c_row`: scaled norms `[1, kv_len]`
// qual:allow(TQ-003) — helper for compute_logit_bias, tested through TqCache integration tests
fn unpack_qjl_signs(
    qjl_signs: &Tensor,
    qjl_norms: &Tensor,
    kv_head: usize,
    total_seq: usize,
    head_dim: usize,
    bit_masks: &Tensor,
    scale_factor: f32,
) -> Result<(Tensor, Tensor)> {
    let head_signs = qjl_signs
        .narrow(0, kv_head, 1)?
        .narrow(1, 0, total_seq)?
        .squeeze(0)?;
    let head_norms = qjl_norms
        .narrow(0, kv_head, 1)?
        .narrow(1, 0, total_seq)?
        .squeeze(0)?
        .to_dtype(DType::F32)?;

    // Unpack U8 signs to ±1.0 float: extract each bit via floor(byte/mask) mod 2.
    // Work in F32 since candle U32 lacks scalar arithmetic and modulo.
    let signs_u8 = head_signs.unsqueeze(2)?;
    let bytes_f = signs_u8.to_dtype(DType::F32)?;
    let masks_f = bit_masks.to_dtype(DType::F32)?;
    let divided = bytes_f.broadcast_div(&masks_f)?.floor()?;
    let bit_set = ((&divided / 2.0)?.floor()? * 2.0 - &divided)?.abs()?;
    let signs_float = ((bit_set * 2.0)? - 1.0)?.reshape((total_seq, head_dim))?;
    let signs_float_t = signs_float.t()?; // [dim, kv_len]

    // Scale: c = norm * sqrt(pi/2) / sqrt(dim)
    let c = (head_norms * scale_factor as f64)?;
    let c_row = c.unsqueeze(0)?; // [1, kv_len]

    Ok((signs_float_t, c_row))
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
    let norms = residual
        .sqr()?
        .sum_keepdim(1)?
        .sqrt()?
        .squeeze(1)?
        .to_dtype(DType::F16)?;

    // Signs computed on CPU (hash-based, not GPU-parallelizable).
    // Extract all residual data at once to avoid per-vector narrow+to_vec1 overhead.
    let residual_cpu = residual.to_device(&Device::Cpu)?;
    let all_residual: Vec<f32> = residual_cpu.flatten_all()?.to_vec1()?;
    let mut all_signs = vec![0u8; n_vecs * signs_per_head];
    for vec_idx in 0..n_vecs {
        let row_data = &all_residual[vec_idx * head_dim..(vec_idx + 1) * head_dim];
        let signs = crate::compute_qjl_signs(row_data, head_dim, DEFAULT_QJL_SEED)
            .map_err(super::cache_err)?;
        let start = vec_idx * signs_per_head;
        all_signs[start..start + signs_per_head].copy_from_slice(&signs);
    }

    let signs =
        Tensor::from_vec(all_signs, n_vecs * signs_per_head, &Device::Cpu)?.to_device(&device)?;
    Ok((signs, norms))
}
