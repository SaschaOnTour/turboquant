//! Compressed GPU tensor storage for KV-cache indices and scales.
//!
//! [`CompressedStorage`] manages the per-layer GPU buffers that hold
//! bit-packed quantization indices and per-block scale factors.
//! Handles capacity growth (25% + 128 headroom) and slice-set operations.

use candle_core::{DType, Device, Result, Tensor};

use super::cache_err;
use super::config::{BITS_PER_BYTE, QUANT_BLOCK_SIZE};

/// Quantized K/V tensor references for a single append operation.
pub struct QuantizedKV<'a> {
    pub k_indices: &'a Tensor,
    pub k_scales: &'a Tensor,
    pub v_indices: &'a Tensor,
    pub v_scales: &'a Tensor,
}

/// Per-layer GPU tensor storage for compressed KV-cache data.
///
/// Fields are kept minimal (SRP): only indices, scales, and bookkeeping.
/// QJL data lives in a separate `QjlStorage` struct.
pub struct CompressedStorage {
    pub(crate) num_kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) bits: u8,
    num_layers: usize,
    buf_seq_len: Vec<usize>,
    gpu_k_indices: Vec<Option<Tensor>>,
    gpu_v_indices: Vec<Option<Tensor>>,
    gpu_k_scales: Vec<Option<Tensor>>,
    gpu_v_scales: Vec<Option<Tensor>>,
    gpu_path_active: Vec<bool>,
}

impl CompressedStorage {
    /// Create empty storage for the given configuration.
    pub fn new(num_kv_heads: usize, head_dim: usize, bits: u8, num_layers: usize) -> Self {
        Self {
            num_kv_heads,
            head_dim,
            bits,
            num_layers,
            buf_seq_len: vec![0; num_layers],
            gpu_k_indices: vec![None; num_layers],
            gpu_v_indices: vec![None; num_layers],
            gpu_k_scales: vec![None; num_layers],
            gpu_v_scales: vec![None; num_layers],
            gpu_path_active: vec![false; num_layers],
        }
    }

    /// Current sequence length for a layer.
    pub fn seq_len(&self, layer: usize) -> usize {
        self.buf_seq_len[layer]
    }

    /// Whether the GPU path is active for a layer (has data stored).
    // qual:allow(TQ-003) — tested via cache_storage_tests
    pub fn is_active(&self, layer: usize) -> bool {
        self.gpu_path_active[layer] && self.buf_seq_len[layer] > 0
    }

    /// Packed dimension: bytes per token for indices.
    pub fn packed_dim(&self) -> usize {
        self.head_dim * self.bits as usize / BITS_PER_BYTE
    }

    /// Number of quantization blocks per head_dim vector.
    pub fn num_blocks(&self) -> usize {
        self.head_dim / QUANT_BLOCK_SIZE
    }

    /// Access key indices tensor for a layer (for fused kernel).
    // qual:allow(TQ-003) — tested via cache_storage_tests
    pub fn k_indices(&self, layer: usize) -> Option<&Tensor> {
        self.gpu_k_indices[layer].as_ref()
    }

    /// Access key scales tensor for a layer (for fused kernel).
    // qual:allow(TQ-003) — tested via cache_storage_tests
    pub fn k_scales(&self, layer: usize) -> Option<&Tensor> {
        self.gpu_k_scales[layer].as_ref()
    }

    /// Access value indices tensor for a layer.
    // qual:allow(TQ-003) — tested via cache_storage_tests
    pub fn v_indices(&self, layer: usize) -> Option<&Tensor> {
        self.gpu_v_indices[layer].as_ref()
    }

    /// Access value scales tensor for a layer.
    // qual:allow(TQ-003) — tested via cache_storage_tests
    pub fn v_scales(&self, layer: usize) -> Option<&Tensor> {
        self.gpu_v_scales[layer].as_ref()
    }

    /// Allocated capacity (max seq_len before realloc) for a layer.
    pub fn capacity(&self, layer: usize) -> usize {
        self.gpu_k_indices[layer]
            .as_ref()
            .map_or(0, |t| t.dims()[1])
    }

    /// Ensure buffers have capacity for at least `needed` tokens.
    /// Grows by 25% + 128 tokens headroom (not doubling — saves VRAM).
    pub fn ensure_capacity(&mut self, layer: usize, needed: usize, device: &Device) -> Result<()> {
        let current_cap = self.capacity(layer);
        if current_cap >= needed {
            return Ok(());
        }

        const MIN_HEADROOM: usize = 128;
        let grow = (needed / 4).max(MIN_HEADROOM);
        let new_cap = needed + grow;
        let old_seq = self.buf_seq_len[layer];
        let heads = self.num_kv_heads;
        let packed_dim = self.packed_dim();
        let num_blocks = self.num_blocks();

        let new_ki = Tensor::zeros((heads, new_cap, packed_dim), DType::U8, device)?;
        let new_vi = Tensor::zeros((heads, new_cap, packed_dim), DType::U8, device)?;
        let new_ks = Tensor::zeros((heads, new_cap, num_blocks), DType::F16, device)?;
        let new_vs = Tensor::zeros((heads, new_cap, num_blocks), DType::F16, device)?;

        if old_seq > 0 {
            copy_old_data(&self.gpu_k_indices[layer], &new_ki, old_seq)?;
            copy_old_data(&self.gpu_v_indices[layer], &new_vi, old_seq)?;
            copy_old_data(&self.gpu_k_scales[layer], &new_ks, old_seq)?;
            copy_old_data(&self.gpu_v_scales[layer], &new_vs, old_seq)?;
        }

        self.gpu_k_indices[layer] = Some(new_ki);
        self.gpu_v_indices[layer] = Some(new_vi);
        self.gpu_k_scales[layer] = Some(new_ks);
        self.gpu_v_scales[layer] = Some(new_vs);
        Ok(())
    }

    /// Append new quantized data at the given offset.
    ///
    /// `k_idx`/`v_idx` shape: `[num_kv_heads, new_seq_len, packed_dim]`
    /// `k_sc`/`v_sc` shape: `[num_kv_heads, new_seq_len, num_blocks]`
    pub fn append(
        &mut self,
        layer: usize,
        offset: usize,
        kv: &QuantizedKV<'_>,
        new_seq_len: usize,
    ) -> Result<()> {
        self.gpu_k_indices[layer]
            .as_ref()
            .ok_or_else(|| cache_err("k_indices buffer not allocated"))?
            .slice_set(kv.k_indices, 1, offset)?;
        self.gpu_v_indices[layer]
            .as_ref()
            .ok_or_else(|| cache_err("v_indices buffer not allocated"))?
            .slice_set(kv.v_indices, 1, offset)?;
        self.gpu_k_scales[layer]
            .as_ref()
            .ok_or_else(|| cache_err("k_scales buffer not allocated"))?
            .slice_set(kv.k_scales, 1, offset)?;
        self.gpu_v_scales[layer]
            .as_ref()
            .ok_or_else(|| cache_err("v_scales buffer not allocated"))?
            .slice_set(kv.v_scales, 1, offset)?;

        self.buf_seq_len[layer] = offset + new_seq_len;
        self.gpu_path_active[layer] = true;
        Ok(())
    }

    /// Reset all layers to empty state.
    // qual:allow(TQ-003) — tested via cache_storage_tests
    pub fn reset(&mut self) {
        for layer in 0..self.num_layers {
            self.gpu_k_indices[layer] = None;
            self.gpu_v_indices[layer] = None;
            self.gpu_k_scales[layer] = None;
            self.gpu_v_scales[layer] = None;
            self.gpu_path_active[layer] = false;
            self.buf_seq_len[layer] = 0;
        }
    }

    /// Estimated persistent memory usage in bytes across all layers.
    pub fn memory_usage(&self) -> usize {
        let mut total = 0;
        for layer in 0..self.num_layers {
            let seq = self.buf_seq_len[layer];
            if seq == 0 {
                continue;
            }
            let packed_dim = self.packed_dim();
            let num_blocks = self.num_blocks();
            // K + V indices (U8) + K + V scales (F16 = 2 bytes)
            total += 2 * self.num_kv_heads * seq * packed_dim;
            total += 2 * self.num_kv_heads * seq * num_blocks * 2;
        }
        total
    }
}

/// Copy old data from existing tensor into new (larger) tensor.
fn copy_old_data(old: &Option<Tensor>, new: &Tensor, old_seq: usize) -> Result<()> {
    if let Some(ref old_tensor) = old {
        let slice = old_tensor.narrow(1, 0, old_seq)?;
        new.slice_set(&slice, 1, 0)?;
    }
    Ok(())
}
