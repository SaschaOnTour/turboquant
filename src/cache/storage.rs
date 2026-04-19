//! Compressed GPU tensor storage for KV-cache indices and scales.
//!
//! [`LayerStorage`] holds one layer's GPU buffers; outer caches wrap it in
//! per-layer locks so different layers can be written concurrently (needed
//! for speculative decoding). [`StorageMetadata`] carries the immutable
//! shape/packing metadata shared across all layers.

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

/// Read-only storage metadata shared across all layers.
#[derive(Clone, Copy)]
pub struct StorageMetadata {
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub bits: u8,
}

impl StorageMetadata {
    /// Packed dimension: bytes per token for indices.
    pub fn packed_dim(&self) -> usize {
        self.head_dim * self.bits as usize / BITS_PER_BYTE
    }

    /// Number of quantization blocks per head_dim vector.
    pub fn num_blocks(&self) -> usize {
        self.head_dim / QUANT_BLOCK_SIZE
    }
}

/// Borrowed view over a layer's four GPU tensor buffers.
///
/// Returned by [`LayerStorage::buffers`] — holds the K/V indices and scales
/// that every decode and dequantize operation reads together.
pub struct LayerBuffers<'a> {
    pub k_indices: &'a Tensor,
    pub k_scales: &'a Tensor,
    pub v_indices: &'a Tensor,
    pub v_scales: &'a Tensor,
}

/// GPU tensor storage for a single transformer layer.
///
/// All per-layer fields are grouped here so an outer cache can wrap one lock
/// per layer (`Mutex<LayerStorage>`) to allow parallel access across layers.
// qual:allow(srp) — cohesive per-layer GPU storage: readers and mutators
// operate on the same (buf_seq_len, gpu_*, gpu_path_active) state. The
// reported LCOM4=2 is a false positive — these fields form one storage
// lifecycle (allocation, growth, reads, GPU-path tracking).
#[derive(Default)]
pub struct LayerStorage {
    pub(crate) buf_seq_len: usize,
    pub(crate) gpu_k_indices: Option<Tensor>,
    pub(crate) gpu_v_indices: Option<Tensor>,
    pub(crate) gpu_k_scales: Option<Tensor>,
    pub(crate) gpu_v_scales: Option<Tensor>,
    pub(crate) gpu_path_active: bool,
}

impl LayerStorage {
    /// Current sequence length.
    pub fn seq_len(&self) -> usize {
        self.buf_seq_len
    }

    /// Whether the GPU path is active (has data stored).
    pub fn is_active(&self) -> bool {
        self.gpu_path_active && self.buf_seq_len > 0
    }

    /// Allocated capacity (max seq_len before realloc).
    pub fn capacity(&self) -> usize {
        self.gpu_k_indices.as_ref().map_or(0, |t| t.dims()[1])
    }

    /// Borrow the four GPU tensors as a group. Returns `None` if any buffer
    /// is not yet allocated (i.e. `ensure_capacity` has not been called).
    pub fn buffers(&self) -> Option<LayerBuffers<'_>> {
        match (
            self.gpu_k_indices.as_ref(),
            self.gpu_k_scales.as_ref(),
            self.gpu_v_indices.as_ref(),
            self.gpu_v_scales.as_ref(),
        ) {
            (Some(ki), Some(ks), Some(vi), Some(vs)) => Some(LayerBuffers {
                k_indices: ki,
                k_scales: ks,
                v_indices: vi,
                v_scales: vs,
            }),
            _ => None,
        }
    }

    /// Ensure buffers have capacity for at least `needed` tokens.
    /// Grows by 25% + 128 tokens headroom (not doubling — saves VRAM).
    pub fn ensure_capacity(
        &mut self,
        needed: usize,
        metadata: &StorageMetadata,
        device: &Device,
    ) -> Result<()> {
        let current_cap = self.capacity();
        if current_cap >= needed {
            return Ok(());
        }

        const MIN_HEADROOM: usize = 128;
        let grow = (needed / 4).max(MIN_HEADROOM);
        let new_cap = needed + grow;
        let heads = metadata.num_kv_heads;
        let packed_dim = metadata.packed_dim();
        let num_blocks = metadata.num_blocks();

        let new_ki = Tensor::zeros((heads, new_cap, packed_dim), DType::U8, device)?;
        let new_vi = Tensor::zeros((heads, new_cap, packed_dim), DType::U8, device)?;
        let new_ks = Tensor::zeros((heads, new_cap, num_blocks), DType::F16, device)?;
        let new_vs = Tensor::zeros((heads, new_cap, num_blocks), DType::F16, device)?;

        let old_seq = self.buf_seq_len;
        if old_seq > 0 {
            copy_old_data(&self.gpu_k_indices, &new_ki, old_seq)?;
            copy_old_data(&self.gpu_v_indices, &new_vi, old_seq)?;
            copy_old_data(&self.gpu_k_scales, &new_ks, old_seq)?;
            copy_old_data(&self.gpu_v_scales, &new_vs, old_seq)?;
        }

        self.gpu_k_indices = Some(new_ki);
        self.gpu_v_indices = Some(new_vi);
        self.gpu_k_scales = Some(new_ks);
        self.gpu_v_scales = Some(new_vs);
        Ok(())
    }

    /// Append new quantized data at the given offset.
    pub fn append(
        &mut self,
        offset: usize,
        kv: &QuantizedKV<'_>,
        new_seq_len: usize,
    ) -> Result<()> {
        self.gpu_k_indices
            .as_ref()
            .ok_or_else(|| cache_err("k_indices buffer not allocated"))?
            .slice_set(kv.k_indices, 1, offset)?;
        self.gpu_v_indices
            .as_ref()
            .ok_or_else(|| cache_err("v_indices buffer not allocated"))?
            .slice_set(kv.v_indices, 1, offset)?;
        self.gpu_k_scales
            .as_ref()
            .ok_or_else(|| cache_err("k_scales buffer not allocated"))?
            .slice_set(kv.k_scales, 1, offset)?;
        self.gpu_v_scales
            .as_ref()
            .ok_or_else(|| cache_err("v_scales buffer not allocated"))?
            .slice_set(kv.v_scales, 1, offset)?;

        self.buf_seq_len = offset + new_seq_len;
        self.gpu_path_active = true;
        debug_assert!(
            self.validate().is_ok(),
            "post-append state must satisfy LayerStorage invariants"
        );
        Ok(())
    }

    /// Verify all internal invariants. Returns an error if the storage is
    /// in an inconsistent state (e.g. active flag disagrees with the buffer
    /// allocation).
    pub fn validate(&self) -> Result<()> {
        if self.gpu_path_active && self.buf_seq_len == 0 {
            return Err(cache_err(
                "active flag set but buf_seq_len is 0 — inconsistent state",
            ));
        }
        if self.gpu_path_active {
            if self.gpu_k_indices.is_none() || self.gpu_v_indices.is_none() {
                return Err(cache_err("active layer missing K/V indices buffer"));
            }
            if self.gpu_k_scales.is_none() || self.gpu_v_scales.is_none() {
                return Err(cache_err("active layer missing K/V scales buffer"));
            }
        }
        let cap = self.capacity();
        if self.buf_seq_len > cap {
            return Err(cache_err(format!(
                "buf_seq_len {} exceeds allocated capacity {}",
                self.buf_seq_len, cap
            )));
        }
        Ok(())
    }

    /// Estimated persistent memory usage in bytes for this layer.
    pub fn memory_usage(&self, metadata: &StorageMetadata) -> usize {
        let seq = self.buf_seq_len;
        if seq == 0 {
            return 0;
        }
        let packed_dim = metadata.packed_dim();
        let num_blocks = metadata.num_blocks();
        // K + V indices (U8) + K + V scales (F16 = 2 bytes)
        2 * metadata.num_kv_heads * seq * packed_dim
            + 2 * metadata.num_kv_heads * seq * num_blocks * 2
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
