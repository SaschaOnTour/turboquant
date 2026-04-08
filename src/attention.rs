//! High-level quantized KV-cache attention API.
//!
//! Provides [`QuantizedKVCache`] for storing quantized key-value pairs
//! and computing attention scores with QJL bias correction.
//!
//! This is the main entry point for users who want to quantize and query
//! KV-caches in a transformer model.

use half::f16;

use crate::codebook::get_codebook;
use crate::error::{check_values_match, require, Result, TurboQuantError};
use crate::packed::{PackedBlock, TurboQuantConfig};
use crate::qjl::{
    estimate_inner_product_with_codebook, precompute_query_projections, quantize_with_qjl,
    EstimationContext, QjlBlock,
};
use crate::quantize::{dequantize_into_with_codebook, DequantScratch};
use crate::rotation::generate_sign_pattern;

// ---------------------------------------------------------------------------
// Named constants (no magic numbers)
// ---------------------------------------------------------------------------

/// Number of bytes per FP16 value.
const BYTES_PER_F16: usize = 2;

/// Number of KV components (key + value).
const KV_PAIR_COUNT: usize = 2;

/// Size of the f16 residual_norm field in a QjlBlock.
const RESIDUAL_NORM_BYTES: usize = 2;

// ---------------------------------------------------------------------------
// QuantizedKVCache
// ---------------------------------------------------------------------------

/// A quantized key-value cache for transformer attention layers.
///
/// Stores keys and values quantized via TURBOQUANTprod (PolarQuant + QJL).
/// Keys are queried via approximate inner products; values are dequantized
/// for weighted summation.
// qual:allow(srp) — cohesive KV-cache struct: management, attention, and dequantization
// all operate on the same (keys, values, config) state.
pub struct QuantizedKVCache {
    /// Quantization configuration (bits, dim, rotation seed).
    config: TurboQuantConfig,
    /// Seed for the QJL Rademacher matrix.
    qjl_seed: u64,
    /// Quantized key blocks per layer: keys[layer][seq_pos].
    keys: Vec<Vec<QjlBlock>>,
    /// Quantized value blocks per layer: values[layer][seq_pos].
    values: Vec<Vec<QjlBlock>>,
}

// ---------------------------------------------------------------------------
// Pure Operation helpers (no project calls)
// ---------------------------------------------------------------------------

/// Checks whether a layer index is within bounds.
///
/// Pure Operation: comparison only.
fn validate_layer(index: usize, num_layers: usize) -> bool {
    index < num_layers
}

/// Fused multiply-add: result[i] += values[i] * weight.
///
/// Pure Operation: arithmetic only.
fn accumulate_weighted(result: &mut [f32], values: &[f32], weight: f32) {
    for (r, &v) in result.iter_mut().zip(values.iter()) {
        *r += v * weight;
    }
}

/// Computes the memory footprint of a single QjlBlock in bytes.
///
/// Pure Operation: field access and arithmetic only.
// iosp:allow — trivial size aggregation: arithmetic between accessor calls
fn qjl_block_size_bytes(block: &QjlBlock) -> usize {
    block.polar_block.size_bytes() + block.qjl_signs.len() + RESIDUAL_NORM_BYTES
}

/// Returns the length of a layer's key vector.
///
/// Pure Operation: field access only.
fn layer_len(layer_keys: &[QjlBlock]) -> usize {
    layer_keys.len()
}

// ---------------------------------------------------------------------------
// Validation helpers (Pure Integration: only calls require + pure ops)
// ---------------------------------------------------------------------------

/// Validates that a layer index is within bounds.
fn check_layer(index: usize, num_layers: usize) -> Result<()> {
    require(
        validate_layer(index, num_layers),
        TurboQuantError::LayerOutOfRange { index, num_layers },
    )
}

/// Validates that all keys and values in a batch have the expected dimension.
fn validate_batch_dims(keys: &[&[f32]], values: &[&[f32]], dim: usize) -> Result<()> {
    for (i, key) in keys.iter().enumerate() {
        check_values_match(key.len(), dim).map_err(|_| TurboQuantError::DimensionMismatch {
            expected: dim,
            actual: key.len(),
        })?;
        check_values_match(values[i].len(), dim).map_err(|_| {
            TurboQuantError::DimensionMismatch {
                expected: dim,
                actual: values[i].len(),
            }
        })?;
    }
    Ok(())
}

/// Checks whether a range `[start..end)` is valid for a given entry count.
fn validate_range(start: usize, end: usize, entry_count: usize) -> bool {
    start <= end && end <= entry_count
}

/// Validates that `[start..end)` is within `[0..entry_count)`.
///
/// Pure Integration: only calls `require` and `validate_range`.
fn check_range(start: usize, end: usize, entry_count: usize) -> Result<()> {
    require(
        validate_range(start, end, entry_count),
        TurboQuantError::RangeOutOfBounds {
            start,
            end,
            entry_count,
        },
    )
}

// ---------------------------------------------------------------------------
// GPU import parameter struct
// ---------------------------------------------------------------------------

/// Parameters for [`QuantizedKVCache::import_packed_range`].
///
/// Groups the many arguments needed to reconstruct QjlBlocks from
/// GPU-quantized flat buffers into a single struct.
pub struct PackedImport<'a> {
    /// Target layer index.
    pub layer: usize,
    /// Polar block bit width (2 for TQ3, 3 for TQ4).
    pub polar_bits: u8,
    /// Concatenated polar block packed indices.
    pub packed_bytes: &'a [u8],
    /// Polar block scale factors as raw u16 (f16 bits).
    pub scales: &'a [u16],
    /// Concatenated QJL sign bytes.
    pub qjl_signs_flat: &'a [u8],
    /// Residual L2 norms as raw u16 (f16 bits).
    pub residual_norms: &'a [u16],
    /// Number of packed bytes per polar block.
    pub bytes_per_block: usize,
    /// Number of QJL sign bytes per block (= ceil(dim/8)).
    pub signs_per_block: usize,
    /// `true` for keys, `false` for values.
    pub is_keys: bool,
}

// ---------------------------------------------------------------------------
// Pure Operation helpers for GPU export/import
// ---------------------------------------------------------------------------

/// Collects packed polar block bytes and f16 scales from a slice of QjlBlocks.
///
/// Pure Operation: accessor calls and collection only.
fn collect_packed_data(blocks: &[QjlBlock]) -> (Vec<u8>, Vec<u16>) {
    let packed_bytes: Vec<u8> = blocks
        .iter()
        .flat_map(|b| b.polar_block().packed_indices())
        .copied()
        .collect();
    let scales: Vec<u16> = blocks
        .iter()
        .map(|b| b.polar_block().scale().to_bits())
        .collect();
    (packed_bytes, scales)
}

/// Reconstructs a single QjlBlock from flat packed buffers at a given index.
///
/// Pure Operation: slicing and construction only.
fn reconstruct_block(import: &PackedImport<'_>, index: usize) -> QjlBlock {
    let pb_start = index * import.bytes_per_block;
    let pb_end = pb_start + import.bytes_per_block;
    let polar_block = PackedBlock::from_raw(
        import.polar_bits,
        f16::from_bits(import.scales[index]),
        import.packed_bytes[pb_start..pb_end].to_vec(),
    );

    let qs_start = index * import.signs_per_block;
    let qs_end = qs_start + import.signs_per_block;
    let qjl_signs = import.qjl_signs_flat[qs_start..qs_end].to_vec();

    let residual_norm = f16::from_bits(import.residual_norms[index]);

    QjlBlock::from_parts(polar_block, qjl_signs, residual_norm)
}

// ---------------------------------------------------------------------------
// Integration: QuantizedKVCache methods
// ---------------------------------------------------------------------------

impl QuantizedKVCache {
    /// Creates a new empty KV cache with the given number of layers.
    ///
    /// Integration: allocates layer vectors.
    pub fn new(config: TurboQuantConfig, num_layers: usize, qjl_seed: u64) -> Self {
        let keys = (0..num_layers).map(|_| Vec::new()).collect();
        let values = (0..num_layers).map(|_| Vec::new()).collect();
        Self {
            config,
            qjl_seed,
            keys,
            values,
        }
    }

    /// Appends a quantized key-value pair to the specified layer.
    ///
    /// Integration: validates layer and dimensions, then delegates to
    /// `quantize_with_qjl` for both key and value.
    ///
    /// # Errors
    ///
    /// Returns [`TurboQuantError::LayerOutOfRange`] if the layer index is invalid.
    /// Returns [`TurboQuantError::DimensionMismatch`] if key or value length
    /// does not match `config.dim`.
    pub fn push(&mut self, layer: usize, key: &[f32], value: &[f32]) -> Result<()> {
        check_layer(layer, self.num_layers())?;
        check_values_match(key.len(), self.config.dim)?;
        check_values_match(value.len(), self.config.dim)?;

        let key_block = quantize_with_qjl(&self.config, key, self.qjl_seed)?;
        let value_block = quantize_with_qjl(&self.config, value, self.qjl_seed)?;

        self.keys[layer].push(key_block);
        self.values[layer].push(value_block);

        Ok(())
    }

    /// Appends multiple quantized key-value pairs to the specified layer in a
    /// single batch.
    ///
    /// More efficient than calling [`push`](Self::push) repeatedly because the
    /// codebook, sign pattern, polar config, and scratch buffers are computed
    /// once and reused across all pairs.  Best used during prefill where
    /// `new_seq_len > 1`.
    ///
    /// # Arguments
    ///
    /// * `layer` - Target layer index.
    /// * `keys`  - Slice of key vectors (each of length `config.dim`).
    /// * `values` - Slice of value vectors (each of length `config.dim`).
    ///
    /// `keys` and `values` must have the same length.
    ///
    /// # Errors
    ///
    /// Returns [`TurboQuantError::LayerOutOfRange`] if the layer index is invalid.
    /// Returns [`TurboQuantError::DimensionMismatch`] if any key or value length
    /// does not match `config.dim`.
    // qual:api — public API for batch insertion during prefill (used by mistral.rs)
    pub fn push_batch(&mut self, layer: usize, keys: &[&[f32]], values: &[&[f32]]) -> Result<()> {
        check_layer(layer, self.num_layers())?;
        check_values_match(keys.len(), values.len())?;
        validate_batch_dims(keys, values, self.config.dim)?;
        self.quantize_and_store(layer, keys, values)
    }

    fn quantize_and_store(
        &mut self,
        layer: usize,
        keys: &[&[f32]],
        values: &[&[f32]],
    ) -> Result<()> {
        use crate::qjl::{quantize_with_qjl_resources, QjlBatchResources};

        let mut res = QjlBatchResources::new(&self.config)?;
        self.keys[layer].reserve(keys.len());
        self.values[layer].reserve(values.len());

        for (key, value) in keys.iter().zip(values.iter()) {
            let key_block = quantize_with_qjl_resources(key, self.qjl_seed, &mut res)?;
            let value_block = quantize_with_qjl_resources(value, self.qjl_seed, &mut res)?;
            self.keys[layer].push(key_block);
            self.values[layer].push(value_block);
        }

        Ok(())
    }

    /// Computes approximate attention scores for a query against all stored keys
    /// in the specified layer.
    ///
    /// Returns a vector of estimated inner products, one per stored key.
    ///
    /// Pre-computes R*query ONCE via `precompute_query_projections`, then
    /// reuses it across all keys via `estimate_inner_product`. Uses
    /// SIMD-friendly sign unpacking.
    ///
    /// Integration: validates layer, pre-computes query projections, then
    /// delegates to `estimate_inner_product` for each key.
    ///
    /// # Errors
    ///
    /// Returns [`TurboQuantError::LayerOutOfRange`] if the layer index is invalid.
    // qual:api — public API for cache consumers (mistral.rs integration)
    pub fn attention_scores(&self, layer: usize, query: &[f32]) -> Result<Vec<f32>> {
        check_layer(layer, self.num_layers())?;

        // Pre-compute R*query ONCE for all keys in the layer.
        let r_query = precompute_query_projections(query, self.config.dim, self.qjl_seed);

        // Fetch codebook, sign pattern, and polar config ONCE before the loop.
        let fetch_and_score = |keys: &[QjlBlock]| -> Result<Vec<f32>> {
            if keys.is_empty() {
                return Ok(Vec::new());
            }
            let polar_bits = keys[0].polar_block.bits();
            let polar_config = TurboQuantConfig::new(polar_bits, self.config.dim)?
                .with_seed(self.config.rotation_seed);
            let codebook = get_codebook(polar_bits, self.config.dim)?;
            let sign_pattern = generate_sign_pattern(self.config.dim, self.config.rotation_seed);

            let mut ctx = EstimationContext {
                polar_config: &polar_config,
                codebook: &codebook,
                sign_pattern: &sign_pattern,
                dim: self.config.dim,
                scratch: DequantScratch::new(self.config.dim),
            };

            let mut scores = Vec::with_capacity(keys.len());
            for key_block in keys {
                let score =
                    estimate_inner_product_with_codebook(query, &r_query, key_block, &mut ctx)?;
                scores.push(score);
            }
            Ok(scores)
        };
        fetch_and_score(&self.keys[layer])
    }

    /// Computes a weighted sum of dequantized values in the specified layer.
    ///
    /// Each value is fully dequantized (with inverse rotation) before
    /// accumulation, because summed values require the original domain.
    /// The polar block uses `(bits-1)` bits, so we create the appropriate
    /// config from each block's `polar_block.bits()`.
    ///
    /// Integration: validates layer and weights length, then delegates to
    /// `dequantize_vec` and `accumulate_weighted`.
    ///
    /// # Errors
    ///
    /// Returns [`TurboQuantError::LayerOutOfRange`] if the layer index is invalid.
    /// Returns [`TurboQuantError::DimensionMismatch`] if `weights.len()` does
    /// not match the number of entries in the layer.
    // qual:api — public API for cache consumers (mistral.rs integration)
    pub fn weighted_values(&self, layer: usize, weights: &[f32]) -> Result<Vec<f32>> {
        check_layer(layer, self.num_layers())?;
        check_values_match(weights.len(), layer_len(&self.keys[layer]))?;

        let dim = self.config.dim;
        let mut result = vec![0.0_f32; dim];

        let dequantize_and_accumulate = |result: &mut Vec<f32>| -> Result<()> {
            let values = &self.values[layer];
            if values.is_empty() {
                return Ok(());
            }
            // Fetch codebook, sign pattern, and polar config ONCE before the loop.
            let polar_bits = values[0].polar_block.bits();
            let polar_config =
                TurboQuantConfig::new(polar_bits, dim)?.with_seed(self.config.rotation_seed);
            let codebook = get_codebook(polar_bits, dim)?;
            let sign_pattern = generate_sign_pattern(dim, self.config.rotation_seed);

            // Pre-allocate scratch buffers reused across all iterations.
            let mut scratch = DequantScratch::new(dim);

            for (block, &w) in values.iter().zip(weights.iter()) {
                dequantize_into_with_codebook(
                    &polar_config,
                    &block.polar_block,
                    &codebook,
                    &sign_pattern,
                    &mut scratch,
                )?;
                accumulate_weighted(result, &scratch.values, w);
            }
            Ok(())
        };
        dequantize_and_accumulate(&mut result)?;

        Ok(result)
    }

    /// Total bytes used by all stored QJL blocks across all layers.
    ///
    /// Integration: delegates to `qjl_block_size_bytes` for each block.
    // qual:api — public API for cache consumers
    pub fn memory_usage(&self) -> usize {
        let key_bytes: usize = self
            .keys
            .iter()
            .flat_map(|layer| layer.iter())
            .map(qjl_block_size_bytes)
            .sum();
        let value_bytes: usize = self
            .values
            .iter()
            .flat_map(|layer| layer.iter())
            .map(qjl_block_size_bytes)
            .sum();
        key_bytes + value_bytes
    }

    /// What the same data would cost in FP16 storage.
    ///
    /// Each entry stores a key and a value, each of dimension `dim`,
    /// at 2 bytes per FP16 element.
    ///
    /// Pure Operation: arithmetic only.
    // qual:api — public API for cache consumers
    pub fn fp16_equivalent_memory(&self) -> usize {
        let total_entries: usize = self.keys.iter().map(|layer| layer.len()).sum();
        total_entries * self.config.dim * BYTES_PER_F16 * KV_PAIR_COUNT
    }

    /// Number of stored entries in a layer.
    ///
    /// Pure Operation: field access only.
    // qual:api — public API for cache consumers
    pub fn entry_count(&self, layer: usize) -> usize {
        layer_len(&self.keys[layer])
    }

    /// Returns a reference to the key QjlBlock at `(layer, index)`.
    ///
    /// Returns `None` if the index is out of bounds.
    ///
    /// Pure Operation: field access only.
    // qual:api — used by mistral.rs TurboQuantKVCache for QJL data access
    pub fn key_block(&self, layer: usize, index: usize) -> Option<&QjlBlock> {
        self.keys.get(layer).and_then(|blocks| blocks.get(index))
    }

    /// Number of layers in the cache.
    ///
    /// Pure Operation: field access only.
    pub fn num_layers(&self) -> usize {
        self.keys.len()
    }

    /// Dequantizes all stored key vectors at the given layer.
    ///
    /// Returns a `Vec` of `Vec<f32>`, one inner vector per stored entry.
    /// Each inner vector has length `config.dim`.
    ///
    /// Integration: validates layer, fetches codebook and sign pattern once,
    /// then delegates to `dequantize_into_with_codebook` for each block.
    ///
    /// # Errors
    ///
    /// Returns [`TurboQuantError::LayerOutOfRange`] if the layer index is invalid.
    // qual:api — used by mistral.rs TurboQuantKVCache integration
    pub fn dequantize_all_keys(&self, layer: usize) -> Result<Vec<Vec<f32>>> {
        self.dequantize_all_blocks(layer, &self.keys)
    }

    /// Dequantizes all stored value vectors at the given layer.
    ///
    /// Returns a `Vec` of `Vec<f32>`, one inner vector per stored entry.
    /// Each inner vector has length `config.dim`.
    ///
    /// Integration: validates layer, fetches codebook and sign pattern once,
    /// then delegates to `dequantize_into_with_codebook` for each block.
    ///
    /// # Errors
    ///
    /// Returns [`TurboQuantError::LayerOutOfRange`] if the layer index is invalid.
    // qual:api — used by mistral.rs TurboQuantKVCache integration
    pub fn dequantize_all_values(&self, layer: usize) -> Result<Vec<Vec<f32>>> {
        self.dequantize_all_blocks(layer, &self.values)
    }

    /// Shared implementation for dequantizing all blocks in a layer.
    ///
    /// Integration: validates layer, fetches codebook and sign pattern once,
    /// then iterates over blocks with scratch-based dequantization.
    fn dequantize_all_blocks(
        &self,
        layer: usize,
        blocks_per_layer: &[Vec<QjlBlock>],
    ) -> Result<Vec<Vec<f32>>> {
        check_layer(layer, self.num_layers())?;
        self.dequantize_block_slice(&blocks_per_layer[layer])
    }

    /// Dequantizes a slice of QjlBlocks into per-entry f32 vectors.
    /// Returns an empty Vec for empty input (safe to call with empty slice).
    // iosp:allow — algorithmic function: codebook/config setup + dequantization loop
    fn dequantize_block_slice(&self, blocks: &[QjlBlock]) -> Result<Vec<Vec<f32>>> {
        if blocks.is_empty() {
            return Ok(Vec::new());
        }
        let dim = self.config.dim;
        let polar_bits = blocks[0].polar_block.bits();
        let polar_config =
            TurboQuantConfig::new(polar_bits, dim)?.with_seed(self.config.rotation_seed);
        let codebook = get_codebook(polar_bits, dim)?;
        let sign_pattern = generate_sign_pattern(dim, self.config.rotation_seed);

        let mut scratch = DequantScratch::new(dim);
        let mut result = Vec::with_capacity(blocks.len());

        for block in blocks {
            dequantize_into_with_codebook(
                &polar_config,
                &block.polar_block,
                &codebook,
                &sign_pattern,
                &mut scratch,
            )?;
            result.push(scratch.values.clone());
        }

        Ok(result)
    }

    /// Shared implementation for dequantizing blocks in a range `[start..end)`
    /// at the given layer.
    ///
    /// Integration: validates layer and range, fetches codebook and sign pattern
    /// once, then iterates only over the requested sub-slice of blocks.
    fn dequantize_blocks_range(
        &self,
        layer: usize,
        start: usize,
        end: usize,
        blocks_per_layer: &[Vec<QjlBlock>],
    ) -> Result<Vec<Vec<f32>>> {
        check_layer(layer, self.num_layers())?;
        let blocks = &blocks_per_layer[layer];
        check_range(start, end, blocks.len())?;
        self.dequantize_block_slice(&blocks[start..end])
    }

    /// Dequantizes key vectors in the range `[start..end)` at the given layer.
    ///
    /// Returns a `Vec` of `Vec<f32>`, one per entry in the range.
    /// Each inner vector has length `config.dim`.
    ///
    /// Integration: validates layer and range, then delegates to
    /// `dequantize_blocks_range`.
    ///
    /// # Errors
    ///
    /// Returns [`TurboQuantError::LayerOutOfRange`] if the layer index is invalid.
    /// Returns [`TurboQuantError::RangeOutOfBounds`] if `start > end` or
    /// `end > entry_count`.
    // qual:api — used by mistral.rs TurboQuantKVCache for delta dequantization
    pub fn dequantize_keys_range(
        &self,
        layer: usize,
        start: usize,
        end: usize,
    ) -> Result<Vec<Vec<f32>>> {
        self.dequantize_blocks_range(layer, start, end, &self.keys)
    }

    /// Dequantizes value vectors in the range `[start..end)` at the given layer.
    ///
    /// Returns a `Vec` of `Vec<f32>`, one per entry in the range.
    /// Each inner vector has length `config.dim`.
    ///
    /// Integration: validates layer and range, then delegates to
    /// `dequantize_blocks_range`.
    ///
    /// # Errors
    ///
    /// Returns [`TurboQuantError::LayerOutOfRange`] if the layer index is invalid.
    /// Returns [`TurboQuantError::RangeOutOfBounds`] if `start > end` or
    /// `end > entry_count`.
    // qual:api — used by mistral.rs TurboQuantKVCache for delta dequantization
    pub fn dequantize_values_range(
        &self,
        layer: usize,
        start: usize,
        end: usize,
    ) -> Result<Vec<Vec<f32>>> {
        self.dequantize_blocks_range(layer, start, end, &self.values)
    }

    // -----------------------------------------------------------------------
    // Block selection helpers (Pure Operation: field access only)
    // -----------------------------------------------------------------------

    /// Returns a reference to the key or value blocks at a given layer.
    fn select_blocks(&self, layer: usize, is_keys: bool) -> &[QjlBlock] {
        if is_keys {
            &self.keys[layer]
        } else {
            &self.values[layer]
        }
    }

    /// Returns a mutable reference to the key or value blocks at a given layer.
    fn select_blocks_mut(&mut self, layer: usize, is_keys: bool) -> &mut Vec<QjlBlock> {
        if is_keys {
            &mut self.keys[layer]
        } else {
            &mut self.values[layer]
        }
    }

    // -----------------------------------------------------------------------
    // GPU integration: export / import packed data
    // -----------------------------------------------------------------------

    /// Returns the quantization configuration.
    ///
    /// Pure Operation: field access only.
    // qual:api — used by GPU kernel integration
    pub fn config(&self) -> &TurboQuantConfig {
        &self.config
    }

    /// Returns the QJL seed used for Rademacher matrix generation.
    ///
    /// Pure Operation: field access only.
    // qual:api — used by GPU kernel integration
    pub fn qjl_seed(&self) -> u64 {
        self.qjl_seed
    }

    /// Exports packed polar block data for a range of entries at a given layer.
    ///
    /// Returns `(flat_packed_bytes, scales_as_u16)` where:
    /// - `flat_packed_bytes` contains all `polar_block.packed_indices()` concatenated
    /// - `scales_as_u16` contains each `polar_block.scale()` as raw `u16` bits
    ///
    /// This is the primary interface for bulk-transferring quantized data to GPU
    /// memory for GPU-side dequantization.
    ///
    /// # Errors
    ///
    /// Returns [`TurboQuantError::LayerOutOfRange`] or
    /// [`TurboQuantError::RangeOutOfBounds`] on invalid arguments.
    // qual:api — used by GPU kernel integration for bulk data export
    pub fn export_packed_range(
        &self,
        layer: usize,
        start: usize,
        end: usize,
        is_keys: bool,
    ) -> Result<(Vec<u8>, Vec<u16>)> {
        check_layer(layer, self.num_layers())?;
        let blocks = self.select_blocks(layer, is_keys);
        check_range(start, end, blocks.len())?;
        Ok(collect_packed_data(&blocks[start..end]))
    }

    /// Imports GPU-quantized data as QjlBlocks into the cache at a given layer.
    ///
    /// See [`PackedImport`] for field documentation.
    ///
    /// # Errors
    ///
    /// Returns [`TurboQuantError::LayerOutOfRange`] on invalid layer.
    // qual:api — used by GPU kernel integration for importing quantized data
    pub fn import_packed_range(&mut self, import: &PackedImport<'_>) -> Result<()> {
        check_layer(import.layer, self.num_layers())?;
        let count = import.scales.len();
        let target = self.select_blocks_mut(import.layer, import.is_keys);
        target.reserve(count);
        for i in 0..count {
            target.push(reconstruct_block(import, i));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qjl::dot_product;
    use crate::quantize::l2_norm;
    use crate::test_utils::{pseudo_random_vec, LCG_MULTIPLIER};

    // -- Named test constants ------------------------------------------------

    /// Test dimension (power of two, suitable for WHT).
    const TEST_DIM: usize = 64;

    /// Bit width for 3-bit quantization.
    const BITS_3: u8 = 3;

    /// Seed for PolarQuant rotation.
    const TEST_ROTATION_SEED: u64 = 42;

    /// Seed for QJL Rademacher matrix.
    const TEST_QJL_SEED: u64 = 12345;

    /// Number of layers in multi-layer tests.
    const TEST_NUM_LAYERS: usize = 4;

    /// Seed for key vector generation.
    const TEST_KEY_SEED: u64 = 11111;

    /// Seed for value vector generation.
    const TEST_VALUE_SEED: u64 = 22222;

    /// Seed for query vector generation.
    const TEST_QUERY_SEED: u64 = 33333;

    /// Second key seed for multi-entry tests.
    const TEST_KEY_SEED_2: u64 = 44444;

    /// Second value seed for multi-entry tests.
    const TEST_VALUE_SEED_2: u64 = 55555;

    /// Third key seed for multi-entry tests.
    const TEST_KEY_SEED_3: u64 = 66666;

    /// Third value seed for multi-entry tests.
    const TEST_VALUE_SEED_3: u64 = 77777;

    /// Maximum relative error for single-vector attention score approximation.
    /// QJL with 3-bit quantization can have ~20% relative error on individual scores.
    const SCORE_RELATIVE_TOLERANCE: f32 = 0.5;

    /// Maximum relative error for weighted value approximation.
    /// Weighted sums accumulate quantization error from multiple values.
    const WEIGHTED_RELATIVE_TOLERANCE: f32 = 1.0;

    /// Number of entries pushed in memory-usage tests.
    const MEMORY_TEST_ENTRIES: usize = 10;

    /// Minimum expected compression ratio for TQ3 (3-bit) vs FP16.
    const MIN_COMPRESSION_RATIO: f32 = 2.0;

    /// Layer index for single-layer tests.
    const TEST_LAYER: usize = 0;

    /// Second layer index for multi-layer tests.
    const TEST_LAYER_2: usize = 1;

    /// An out-of-range layer index.
    const INVALID_LAYER: usize = 999;

    /// Tolerance for floating-point comparisons.
    const FLOAT_EPSILON: f32 = 1e-6;

    /// Weight used in simple weighted-values tests.
    const TEST_WEIGHT_A: f32 = 0.6;

    /// Weight used in simple weighted-values tests.
    const TEST_WEIGHT_B: f32 = 0.4;

    /// Number of entries for FP16-equivalent test.
    const FP16_TEST_ENTRIES: usize = 5;

    /// Seed offset base for multi-entry loops.
    const SEED_OFFSET_BASE: u64 = 100;

    /// Creates a test config with standard parameters.
    fn test_config() -> TurboQuantConfig {
        TurboQuantConfig::new(BITS_3, TEST_DIM)
            .unwrap()
            .with_seed(TEST_ROTATION_SEED)
    }

    // -- Cache push/read roundtrip -------------------------------------------

    #[test]
    fn push_increases_len() {
        let config = test_config();
        let mut cache = QuantizedKVCache::new(config, TEST_NUM_LAYERS, TEST_QJL_SEED);

        assert_eq!(cache.entry_count(TEST_LAYER), 0);

        let key = pseudo_random_vec(TEST_DIM, TEST_KEY_SEED);
        let value = pseudo_random_vec(TEST_DIM, TEST_VALUE_SEED);
        cache.push(TEST_LAYER, &key, &value).unwrap();

        assert_eq!(cache.entry_count(TEST_LAYER), 1);

        let key2 = pseudo_random_vec(TEST_DIM, TEST_KEY_SEED_2);
        let value2 = pseudo_random_vec(TEST_DIM, TEST_VALUE_SEED_2);
        cache.push(TEST_LAYER, &key2, &value2).unwrap();

        assert_eq!(cache.entry_count(TEST_LAYER), 2);
    }

    // -- Attention scores ----------------------------------------------------

    #[test]
    fn attention_scores_approximate_dot_product() {
        let config = test_config();
        let mut cache = QuantizedKVCache::new(config, TEST_NUM_LAYERS, TEST_QJL_SEED);

        let key = pseudo_random_vec(TEST_DIM, TEST_KEY_SEED);
        let value = pseudo_random_vec(TEST_DIM, TEST_VALUE_SEED);
        cache.push(TEST_LAYER, &key, &value).unwrap();

        let query = pseudo_random_vec(TEST_DIM, TEST_QUERY_SEED);
        let scores = cache.attention_scores(TEST_LAYER, &query).unwrap();

        assert_eq!(scores.len(), 1);

        let true_ip = dot_product(&query, &key);
        let relative_error = (scores[0] - true_ip).abs() / true_ip.abs().max(1.0);
        assert!(
            relative_error < SCORE_RELATIVE_TOLERANCE,
            "score relative error {relative_error} exceeds tolerance {SCORE_RELATIVE_TOLERANCE} \
             (estimated={}, true={true_ip})",
            scores[0]
        );
    }

    // -- Weighted values -----------------------------------------------------

    #[test]
    fn weighted_values_approximate_naive_sum() {
        let config = test_config();
        let mut cache = QuantizedKVCache::new(config, TEST_NUM_LAYERS, TEST_QJL_SEED);

        let key1 = pseudo_random_vec(TEST_DIM, TEST_KEY_SEED);
        let val1 = pseudo_random_vec(TEST_DIM, TEST_VALUE_SEED);
        cache.push(TEST_LAYER, &key1, &val1).unwrap();

        let key2 = pseudo_random_vec(TEST_DIM, TEST_KEY_SEED_2);
        let val2 = pseudo_random_vec(TEST_DIM, TEST_VALUE_SEED_2);
        cache.push(TEST_LAYER, &key2, &val2).unwrap();

        let weights = vec![TEST_WEIGHT_A, TEST_WEIGHT_B];
        let result = cache.weighted_values(TEST_LAYER, &weights).unwrap();

        // Compute naive weighted sum from original values
        let naive: Vec<f32> = (0..TEST_DIM)
            .map(|i| val1[i] * TEST_WEIGHT_A + val2[i] * TEST_WEIGHT_B)
            .collect();

        let error_vec: Vec<f32> = result
            .iter()
            .zip(naive.iter())
            .map(|(&r, &n)| r - n)
            .collect();
        let error_norm = l2_norm(&error_vec);
        let naive_norm = l2_norm(&naive);
        let relative_error = error_norm / naive_norm;

        assert!(
            relative_error < WEIGHTED_RELATIVE_TOLERANCE,
            "weighted values relative error {relative_error} exceeds tolerance {WEIGHTED_RELATIVE_TOLERANCE}"
        );
    }

    // -- Memory usage --------------------------------------------------------

    #[test]
    fn memory_usage_shows_compression() {
        let config = test_config();
        let mut cache = QuantizedKVCache::new(config, TEST_NUM_LAYERS, TEST_QJL_SEED);

        for i in 0..MEMORY_TEST_ENTRIES {
            let key = pseudo_random_vec(
                TEST_DIM,
                TEST_KEY_SEED.wrapping_add(i as u64 * SEED_OFFSET_BASE),
            );
            let val = pseudo_random_vec(
                TEST_DIM,
                TEST_VALUE_SEED.wrapping_add(i as u64 * SEED_OFFSET_BASE),
            );
            cache.push(TEST_LAYER, &key, &val).unwrap();
        }

        let quantized_bytes = cache.memory_usage();
        let fp16_bytes = cache.fp16_equivalent_memory();

        assert!(quantized_bytes > 0);
        assert!(fp16_bytes > 0);

        let ratio = fp16_bytes as f32 / quantized_bytes as f32;
        assert!(
            ratio > MIN_COMPRESSION_RATIO,
            "compression ratio {ratio} is below minimum {MIN_COMPRESSION_RATIO}"
        );
    }

    // -- FP16 equivalent memory ----------------------------------------------

    #[test]
    fn fp16_equivalent_memory_correct_calculation() {
        let config = test_config();
        let mut cache = QuantizedKVCache::new(config, TEST_NUM_LAYERS, TEST_QJL_SEED);

        for i in 0..FP16_TEST_ENTRIES {
            let key = pseudo_random_vec(
                TEST_DIM,
                TEST_KEY_SEED_3.wrapping_add(i as u64 * SEED_OFFSET_BASE),
            );
            let val = pseudo_random_vec(
                TEST_DIM,
                TEST_VALUE_SEED_3.wrapping_add(i as u64 * SEED_OFFSET_BASE),
            );
            cache.push(TEST_LAYER, &key, &val).unwrap();
        }

        let expected = FP16_TEST_ENTRIES * TEST_DIM * BYTES_PER_F16 * KV_PAIR_COUNT;
        assert_eq!(cache.fp16_equivalent_memory(), expected);
    }

    // -- Multi-layer cache ---------------------------------------------------

    #[test]
    fn multi_layer_cache_independent() {
        let config = test_config();
        let mut cache = QuantizedKVCache::new(config, TEST_NUM_LAYERS, TEST_QJL_SEED);

        let key1 = pseudo_random_vec(TEST_DIM, TEST_KEY_SEED);
        let val1 = pseudo_random_vec(TEST_DIM, TEST_VALUE_SEED);
        cache.push(TEST_LAYER, &key1, &val1).unwrap();

        let key2 = pseudo_random_vec(TEST_DIM, TEST_KEY_SEED_2);
        let val2 = pseudo_random_vec(TEST_DIM, TEST_VALUE_SEED_2);
        cache.push(TEST_LAYER_2, &key2, &val2).unwrap();
        cache.push(TEST_LAYER_2, &key2, &val2).unwrap();

        assert_eq!(cache.entry_count(TEST_LAYER), 1);
        assert_eq!(cache.entry_count(TEST_LAYER_2), 2);
        assert_eq!(cache.num_layers(), TEST_NUM_LAYERS);
    }

    // -- Empty cache ---------------------------------------------------------

    #[test]
    fn empty_cache_returns_empty_scores() {
        let config = test_config();
        let cache = QuantizedKVCache::new(config, TEST_NUM_LAYERS, TEST_QJL_SEED);
        let query = pseudo_random_vec(TEST_DIM, TEST_QUERY_SEED);
        let scores = cache.attention_scores(TEST_LAYER, &query).unwrap();
        assert!(scores.is_empty());
    }

    // -- Layer validation ----------------------------------------------------

    #[test]
    fn push_rejects_invalid_layer() {
        let config = test_config();
        let mut cache = QuantizedKVCache::new(config, TEST_NUM_LAYERS, TEST_QJL_SEED);
        let key = pseudo_random_vec(TEST_DIM, TEST_KEY_SEED);
        let val = pseudo_random_vec(TEST_DIM, TEST_VALUE_SEED);
        let result = cache.push(INVALID_LAYER, &key, &val);
        assert!(result.is_err());
    }

    #[test]
    fn attention_scores_rejects_invalid_layer() {
        let config = test_config();
        let cache = QuantizedKVCache::new(config, TEST_NUM_LAYERS, TEST_QJL_SEED);
        let query = pseudo_random_vec(TEST_DIM, TEST_QUERY_SEED);
        let result = cache.attention_scores(INVALID_LAYER, &query);
        assert!(result.is_err());
    }

    #[test]
    fn weighted_values_rejects_invalid_layer() {
        let config = test_config();
        let cache = QuantizedKVCache::new(config, TEST_NUM_LAYERS, TEST_QJL_SEED);
        let weights = vec![1.0_f32];
        let result = cache.weighted_values(INVALID_LAYER, &weights);
        assert!(result.is_err());
    }

    // -- Dimension validation ------------------------------------------------

    #[test]
    fn push_rejects_wrong_key_dimension() {
        let config = test_config();
        let mut cache = QuantizedKVCache::new(config, TEST_NUM_LAYERS, TEST_QJL_SEED);
        let wrong_key = vec![1.0_f32; TEST_DIM + 1];
        let val = pseudo_random_vec(TEST_DIM, TEST_VALUE_SEED);
        let result = cache.push(TEST_LAYER, &wrong_key, &val);
        assert!(result.is_err());
    }

    #[test]
    fn push_rejects_wrong_value_dimension() {
        let config = test_config();
        let mut cache = QuantizedKVCache::new(config, TEST_NUM_LAYERS, TEST_QJL_SEED);
        let key = pseudo_random_vec(TEST_DIM, TEST_KEY_SEED);
        let wrong_val = vec![1.0_f32; TEST_DIM + 1];
        let result = cache.push(TEST_LAYER, &key, &wrong_val);
        assert!(result.is_err());
    }

    // -- Pure operation helpers ----------------------------------------------

    #[test]
    fn validate_layer_in_bounds() {
        assert!(validate_layer(0, TEST_NUM_LAYERS));
        assert!(validate_layer(TEST_NUM_LAYERS - 1, TEST_NUM_LAYERS));
    }

    #[test]
    fn validate_layer_out_of_bounds() {
        assert!(!validate_layer(TEST_NUM_LAYERS, TEST_NUM_LAYERS));
        assert!(!validate_layer(INVALID_LAYER, TEST_NUM_LAYERS));
    }

    #[test]
    fn accumulate_weighted_basic() {
        let mut result = vec![0.0_f32; TEST_DIM];
        let values: Vec<f32> = (0..TEST_DIM).map(|i| i as f32).collect();
        let weight = TEST_WEIGHT_A;

        accumulate_weighted(&mut result, &values, weight);

        for (i, &r) in result.iter().enumerate() {
            let expected = i as f32 * weight;
            assert!(
                (r - expected).abs() < FLOAT_EPSILON,
                "mismatch at index {i}: expected {expected}, got {r}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Weighted values full roundtrip quality test
    // -----------------------------------------------------------------------

    /// Dimension for weighted values roundtrip test.
    const WV_ROUNDTRIP_DIM: usize = 128;

    /// Number of KV pairs for weighted values roundtrip test.
    const WV_ROUNDTRIP_ENTRIES: usize = 50;

    /// Seed offset for weighted values roundtrip key generation.
    const WV_KEY_SEED_OFFSET: u64 = 90000;

    /// Seed offset for weighted values roundtrip value generation.
    const WV_VALUE_SEED_OFFSET: u64 = 91000;

    /// Uniform weight for each entry (1/50).
    const WV_UNIFORM_WEIGHT: f32 = 1.0 / WV_ROUNDTRIP_ENTRIES as f32;

    /// Maximum acceptable relative error for uniform-weighted value roundtrip.
    /// Loose bound (< 1.0) but catches gross reconstruction bugs.
    const WV_MAX_RELATIVE_ERROR: f32 = 1.0;

    /// Rotation seed for weighted values roundtrip test.
    const WV_ROTATION_SEED: u64 = 42;

    /// QJL seed for weighted values roundtrip test.
    const WV_QJL_SEED: u64 = 31415;

    #[test]
    fn weighted_values_uniform_roundtrip_quality() {
        let config = TurboQuantConfig::new(BITS_3, WV_ROUNDTRIP_DIM)
            .unwrap()
            .with_seed(WV_ROTATION_SEED);
        let mut cache = QuantizedKVCache::new(config, 1, WV_QJL_SEED);

        // Store original values for ground-truth comparison.
        let mut original_values: Vec<Vec<f32>> = Vec::with_capacity(WV_ROUNDTRIP_ENTRIES);

        for i in 0..WV_ROUNDTRIP_ENTRIES {
            let key_seed = (i as u64)
                .wrapping_mul(LCG_MULTIPLIER)
                .wrapping_add(WV_KEY_SEED_OFFSET);
            let val_seed = (i as u64)
                .wrapping_mul(LCG_MULTIPLIER)
                .wrapping_add(WV_VALUE_SEED_OFFSET);

            let key = pseudo_random_vec(WV_ROUNDTRIP_DIM, key_seed);
            let val = pseudo_random_vec(WV_ROUNDTRIP_DIM, val_seed);

            original_values.push(val.clone());
            cache.push(0, &key, &val).unwrap();
        }

        // Uniform weights: 1/N for each entry.
        let weights = vec![WV_UNIFORM_WEIGHT; WV_ROUNDTRIP_ENTRIES];
        let result = cache.weighted_values(0, &weights).unwrap();

        assert_eq!(result.len(), WV_ROUNDTRIP_DIM);

        // Compute naive average of original values.
        let mut naive_avg = vec![0.0_f32; WV_ROUNDTRIP_DIM];
        for val in &original_values {
            for (j, &v) in val.iter().enumerate() {
                naive_avg[j] += v * WV_UNIFORM_WEIGHT;
            }
        }

        // Compute relative error between quantized weighted sum and naive average.
        let error_vec: Vec<f32> = result
            .iter()
            .zip(naive_avg.iter())
            .map(|(&r, &n)| r - n)
            .collect();
        let error_norm = l2_norm(&error_vec);
        let naive_norm = l2_norm(&naive_avg);

        const NORM_FLOOR: f32 = 1e-10;
        let relative_error = error_norm / naive_norm.max(NORM_FLOOR);
        eprintln!(
            "Weighted values uniform roundtrip: relative_error={relative_error:.4}, \
             error_norm={error_norm:.6}, naive_norm={naive_norm:.6}"
        );
        assert!(
            relative_error < WV_MAX_RELATIVE_ERROR,
            "Weighted values uniform roundtrip relative error {relative_error:.4} \
             exceeds tolerance {WV_MAX_RELATIVE_ERROR}"
        );
    }

    // -----------------------------------------------------------------------
    // Compression ratio verification tests
    // -----------------------------------------------------------------------

    /// Dimension used in compression-ratio tests (power of two, realistic head size).
    const RATIO_TEST_DIM: usize = 128;

    /// Number of entries pushed in compression-ratio tests.
    const RATIO_TEST_ENTRIES: usize = 100;

    /// Bit width for TQ4 (4-bit total budget).
    const BITS_4: u8 = 4;

    /// Minimum expected compression ratio for TQ3 (3-bit) vs FP16.
    /// TQ3: 3 bits/dim data + small overhead vs FP16: 16 bits/dim.
    /// Per KV pair: quantized ~52 bytes * 2 = 104 vs FP16 = 512 bytes.
    /// Expected >= 4.0x.
    const TQ3_MIN_COMPRESSION_RATIO: f32 = 4.0;

    /// Minimum expected compression ratio for TQ4 (4-bit) vs FP16.
    /// TQ4: 4 bits/dim data + small overhead vs FP16: 16 bits/dim.
    /// Expected >= 3.0x.
    const TQ4_MIN_COMPRESSION_RATIO: f32 = 3.0;

    /// Seed offset for compression-ratio test entry generation.
    const RATIO_SEED_OFFSET: u64 = 7000;

    #[test]
    fn tq3_compression_ratio_meets_minimum() {
        let config = TurboQuantConfig::new(BITS_3, RATIO_TEST_DIM)
            .unwrap()
            .with_seed(TEST_ROTATION_SEED);
        let mut cache = QuantizedKVCache::new(config, 1, TEST_QJL_SEED);

        for i in 0..RATIO_TEST_ENTRIES {
            let key = pseudo_random_vec(
                RATIO_TEST_DIM,
                TEST_KEY_SEED.wrapping_add(i as u64 * RATIO_SEED_OFFSET),
            );
            let val = pseudo_random_vec(
                RATIO_TEST_DIM,
                TEST_VALUE_SEED.wrapping_add(i as u64 * RATIO_SEED_OFFSET),
            );
            cache.push(0, &key, &val).unwrap();
        }

        let quantized_bytes = cache.memory_usage();
        let fp16_bytes = cache.fp16_equivalent_memory();
        let ratio = fp16_bytes as f32 / quantized_bytes as f32;

        assert!(
            ratio >= TQ3_MIN_COMPRESSION_RATIO,
            "TQ3 compression ratio {ratio:.2}x is below minimum {TQ3_MIN_COMPRESSION_RATIO}x \
             (quantized={quantized_bytes} bytes, fp16={fp16_bytes} bytes)"
        );
    }

    // -----------------------------------------------------------------------
    // Range dequantization tests
    // -----------------------------------------------------------------------

    /// Number of entries for range dequantization tests.
    const RANGE_TEST_ENTRIES: usize = 5;

    /// Seed offset for range test entry generation.
    const RANGE_SEED_OFFSET: u64 = 500;

    /// Populates a cache with `RANGE_TEST_ENTRIES` entries at `TEST_LAYER`.
    fn make_range_test_cache() -> QuantizedKVCache {
        let config = test_config();
        let mut cache = QuantizedKVCache::new(config, TEST_NUM_LAYERS, TEST_QJL_SEED);
        for i in 0..RANGE_TEST_ENTRIES {
            let key = pseudo_random_vec(
                TEST_DIM,
                TEST_KEY_SEED.wrapping_add(i as u64 * RANGE_SEED_OFFSET),
            );
            let val = pseudo_random_vec(
                TEST_DIM,
                TEST_VALUE_SEED.wrapping_add(i as u64 * RANGE_SEED_OFFSET),
            );
            cache.push(TEST_LAYER, &key, &val).unwrap();
        }
        cache
    }

    #[test]
    fn dequantize_keys_range_full_matches_all() {
        let cache = make_range_test_cache();
        let all = cache.dequantize_all_keys(TEST_LAYER).unwrap();
        let range_all = cache
            .dequantize_keys_range(TEST_LAYER, 0, RANGE_TEST_ENTRIES)
            .unwrap();
        assert_eq!(all.len(), range_all.len());
        for (a, r) in all.iter().zip(range_all.iter()) {
            assert_eq!(a, r);
        }
    }

    #[test]
    fn dequantize_values_range_full_matches_all() {
        let cache = make_range_test_cache();
        let all = cache.dequantize_all_values(TEST_LAYER).unwrap();
        let range_all = cache
            .dequantize_values_range(TEST_LAYER, 0, RANGE_TEST_ENTRIES)
            .unwrap();
        assert_eq!(all.len(), range_all.len());
        for (a, r) in all.iter().zip(range_all.iter()) {
            assert_eq!(a, r);
        }
    }

    #[test]
    fn dequantize_keys_range_subset_matches_slice() {
        let cache = make_range_test_cache();
        let all = cache.dequantize_all_keys(TEST_LAYER).unwrap();
        let start = 1;
        let end = 3;
        let range_subset = cache.dequantize_keys_range(TEST_LAYER, start, end).unwrap();
        assert_eq!(range_subset.len(), end - start);
        for (i, vec) in range_subset.iter().enumerate() {
            assert_eq!(vec, &all[start + i]);
        }
    }

    #[test]
    fn dequantize_values_range_subset_matches_slice() {
        let cache = make_range_test_cache();
        let all = cache.dequantize_all_values(TEST_LAYER).unwrap();
        let start = 2;
        let end = 4;
        let range_subset = cache
            .dequantize_values_range(TEST_LAYER, start, end)
            .unwrap();
        assert_eq!(range_subset.len(), end - start);
        for (i, vec) in range_subset.iter().enumerate() {
            assert_eq!(vec, &all[start + i]);
        }
    }

    #[test]
    fn dequantize_keys_range_empty_returns_empty() {
        let cache = make_range_test_cache();
        let empty = cache.dequantize_keys_range(TEST_LAYER, 2, 2).unwrap();
        assert!(empty.is_empty());
    }

    #[test]
    fn dequantize_values_range_empty_returns_empty() {
        let cache = make_range_test_cache();
        let empty = cache.dequantize_values_range(TEST_LAYER, 0, 0).unwrap();
        assert!(empty.is_empty());
    }

    #[test]
    fn dequantize_keys_range_single_entry() {
        let cache = make_range_test_cache();
        let all = cache.dequantize_all_keys(TEST_LAYER).unwrap();
        let single = cache.dequantize_keys_range(TEST_LAYER, 3, 4).unwrap();
        assert_eq!(single.len(), 1);
        assert_eq!(single[0], all[3]);
    }

    #[test]
    fn dequantize_keys_range_rejects_invalid_layer() {
        let cache = make_range_test_cache();
        let result = cache.dequantize_keys_range(INVALID_LAYER, 0, 1);
        assert!(result.is_err());
    }

    #[test]
    fn dequantize_values_range_rejects_invalid_layer() {
        let cache = make_range_test_cache();
        let result = cache.dequantize_values_range(INVALID_LAYER, 0, 1);
        assert!(result.is_err());
    }

    #[test]
    fn dequantize_keys_range_rejects_start_greater_than_end() {
        let cache = make_range_test_cache();
        let result = cache.dequantize_keys_range(TEST_LAYER, 3, 1);
        assert!(result.is_err());
    }

    #[test]
    fn dequantize_values_range_rejects_end_beyond_entry_count() {
        let cache = make_range_test_cache();
        let result = cache.dequantize_values_range(TEST_LAYER, 0, RANGE_TEST_ENTRIES + 1);
        assert!(result.is_err());
    }

    #[test]
    fn validate_range_pure_operation() {
        assert!(validate_range(0, 0, 0));
        assert!(validate_range(0, 5, 5));
        assert!(validate_range(2, 3, 5));
        assert!(!validate_range(3, 2, 5));
        assert!(!validate_range(0, 6, 5));
    }

    #[test]
    fn tq4_compression_ratio_meets_minimum() {
        let config = TurboQuantConfig::new(BITS_4, RATIO_TEST_DIM)
            .unwrap()
            .with_seed(TEST_ROTATION_SEED);
        let mut cache = QuantizedKVCache::new(config, 1, TEST_QJL_SEED);

        for i in 0..RATIO_TEST_ENTRIES {
            let key = pseudo_random_vec(
                RATIO_TEST_DIM,
                TEST_KEY_SEED.wrapping_add(i as u64 * RATIO_SEED_OFFSET),
            );
            let val = pseudo_random_vec(
                RATIO_TEST_DIM,
                TEST_VALUE_SEED.wrapping_add(i as u64 * RATIO_SEED_OFFSET),
            );
            cache.push(0, &key, &val).unwrap();
        }

        let quantized_bytes = cache.memory_usage();
        let fp16_bytes = cache.fp16_equivalent_memory();
        let ratio = fp16_bytes as f32 / quantized_bytes as f32;

        assert!(
            ratio >= TQ4_MIN_COMPRESSION_RATIO,
            "TQ4 compression ratio {ratio:.2}x is below minimum {TQ4_MIN_COMPRESSION_RATIO}x \
             (quantized={quantized_bytes} bytes, fp16={fp16_bytes} bytes)"
        );
    }

    // -----------------------------------------------------------------------
    // push_batch tests
    // -----------------------------------------------------------------------

    /// Number of vectors in the batch test.
    const BATCH_TEST_COUNT: usize = 8;

    /// Seed offset for batch test vectors.
    const BATCH_SEED_OFFSET: u64 = 3000;

    #[test]
    fn push_batch_produces_same_results_as_individual_pushes() {
        let config = test_config();

        // Build key/value vectors
        let mut key_vecs: Vec<Vec<f32>> = Vec::new();
        let mut val_vecs: Vec<Vec<f32>> = Vec::new();
        for i in 0..BATCH_TEST_COUNT {
            key_vecs.push(pseudo_random_vec(
                TEST_DIM,
                TEST_KEY_SEED.wrapping_add(i as u64 * BATCH_SEED_OFFSET),
            ));
            val_vecs.push(pseudo_random_vec(
                TEST_DIM,
                TEST_VALUE_SEED.wrapping_add(i as u64 * BATCH_SEED_OFFSET),
            ));
        }

        // Individual pushes
        let mut cache_individual = QuantizedKVCache::new(config, TEST_NUM_LAYERS, TEST_QJL_SEED);
        for i in 0..BATCH_TEST_COUNT {
            cache_individual
                .push(TEST_LAYER, &key_vecs[i], &val_vecs[i])
                .unwrap();
        }

        // Batch push
        let mut cache_batch = QuantizedKVCache::new(config, TEST_NUM_LAYERS, TEST_QJL_SEED);
        let key_refs: Vec<&[f32]> = key_vecs.iter().map(|v| v.as_slice()).collect();
        let val_refs: Vec<&[f32]> = val_vecs.iter().map(|v| v.as_slice()).collect();
        cache_batch
            .push_batch(TEST_LAYER, &key_refs, &val_refs)
            .unwrap();

        // Verify same entry count
        assert_eq!(
            cache_individual.entry_count(TEST_LAYER),
            cache_batch.entry_count(TEST_LAYER),
        );

        // Verify dequantized keys are identical
        let keys_ind = cache_individual.dequantize_all_keys(TEST_LAYER).unwrap();
        let keys_bat = cache_batch.dequantize_all_keys(TEST_LAYER).unwrap();
        assert_eq!(keys_ind.len(), keys_bat.len());
        for (a, b) in keys_ind.iter().zip(keys_bat.iter()) {
            assert_eq!(a, b, "batch and individual key dequantizations differ");
        }

        // Verify dequantized values are identical
        let vals_ind = cache_individual.dequantize_all_values(TEST_LAYER).unwrap();
        let vals_bat = cache_batch.dequantize_all_values(TEST_LAYER).unwrap();
        assert_eq!(vals_ind.len(), vals_bat.len());
        for (a, b) in vals_ind.iter().zip(vals_bat.iter()) {
            assert_eq!(a, b, "batch and individual value dequantizations differ");
        }
    }

    #[test]
    fn push_batch_empty_is_noop() {
        let config = test_config();
        let mut cache = QuantizedKVCache::new(config, TEST_NUM_LAYERS, TEST_QJL_SEED);
        let empty_keys: Vec<&[f32]> = Vec::new();
        let empty_vals: Vec<&[f32]> = Vec::new();
        cache
            .push_batch(TEST_LAYER, &empty_keys, &empty_vals)
            .unwrap();
        assert_eq!(cache.entry_count(TEST_LAYER), 0);
    }

    #[test]
    fn push_batch_rejects_invalid_layer() {
        let config = test_config();
        let mut cache = QuantizedKVCache::new(config, TEST_NUM_LAYERS, TEST_QJL_SEED);
        let key = pseudo_random_vec(TEST_DIM, TEST_KEY_SEED);
        let val = pseudo_random_vec(TEST_DIM, TEST_VALUE_SEED);
        let keys: Vec<&[f32]> = vec![&key];
        let vals: Vec<&[f32]> = vec![&val];
        let result = cache.push_batch(INVALID_LAYER, &keys, &vals);
        assert!(result.is_err());
    }

    #[test]
    fn push_batch_rejects_wrong_dimension() {
        let config = test_config();
        let mut cache = QuantizedKVCache::new(config, TEST_NUM_LAYERS, TEST_QJL_SEED);
        let wrong_key = vec![1.0_f32; TEST_DIM + 1];
        let val = pseudo_random_vec(TEST_DIM, TEST_VALUE_SEED);
        let keys: Vec<&[f32]> = vec![wrong_key.as_slice()];
        let vals: Vec<&[f32]> = vec![val.as_slice()];
        let result = cache.push_batch(TEST_LAYER, &keys, &vals);
        assert!(result.is_err());
    }

    #[test]
    fn push_batch_rejects_mismatched_lengths() {
        let config = test_config();
        let mut cache = QuantizedKVCache::new(config, TEST_NUM_LAYERS, TEST_QJL_SEED);
        let key = pseudo_random_vec(TEST_DIM, TEST_KEY_SEED);
        let val1 = pseudo_random_vec(TEST_DIM, TEST_VALUE_SEED);
        let val2 = pseudo_random_vec(TEST_DIM, TEST_VALUE_SEED_2);
        let keys: Vec<&[f32]> = vec![key.as_slice()];
        let vals: Vec<&[f32]> = vec![val1.as_slice(), val2.as_slice()];
        let result = cache.push_batch(TEST_LAYER, &keys, &vals);
        assert!(result.is_err());
    }

    #[test]
    fn push_batch_single_element_matches_push() {
        let config = test_config();
        let key = pseudo_random_vec(TEST_DIM, TEST_KEY_SEED);
        let val = pseudo_random_vec(TEST_DIM, TEST_VALUE_SEED);

        let mut cache_push = QuantizedKVCache::new(config, TEST_NUM_LAYERS, TEST_QJL_SEED);
        cache_push.push(TEST_LAYER, &key, &val).unwrap();

        let mut cache_batch = QuantizedKVCache::new(config, TEST_NUM_LAYERS, TEST_QJL_SEED);
        let keys: Vec<&[f32]> = vec![key.as_slice()];
        let vals: Vec<&[f32]> = vec![val.as_slice()];
        cache_batch.push_batch(TEST_LAYER, &keys, &vals).unwrap();

        assert_eq!(
            cache_push.entry_count(TEST_LAYER),
            cache_batch.entry_count(TEST_LAYER)
        );

        let k_push = cache_push.dequantize_all_keys(TEST_LAYER).unwrap();
        let k_batch = cache_batch.dequantize_all_keys(TEST_LAYER).unwrap();
        assert_eq!(k_push, k_batch);

        let v_push = cache_push.dequantize_all_values(TEST_LAYER).unwrap();
        let v_batch = cache_batch.dequantize_all_values(TEST_LAYER).unwrap();
        assert_eq!(v_push, v_batch);
    }

    #[test]
    fn push_batch_then_individual_push_consistent() {
        let config = test_config();
        let mut cache = QuantizedKVCache::new(config, TEST_NUM_LAYERS, TEST_QJL_SEED);

        // First: batch push of 3 entries
        let mut key_vecs: Vec<Vec<f32>> = Vec::new();
        let mut val_vecs: Vec<Vec<f32>> = Vec::new();
        for i in 0..3 {
            key_vecs.push(pseudo_random_vec(
                TEST_DIM,
                TEST_KEY_SEED.wrapping_add(i as u64 * BATCH_SEED_OFFSET),
            ));
            val_vecs.push(pseudo_random_vec(
                TEST_DIM,
                TEST_VALUE_SEED.wrapping_add(i as u64 * BATCH_SEED_OFFSET),
            ));
        }
        let key_refs: Vec<&[f32]> = key_vecs.iter().map(|v| v.as_slice()).collect();
        let val_refs: Vec<&[f32]> = val_vecs.iter().map(|v| v.as_slice()).collect();
        cache.push_batch(TEST_LAYER, &key_refs, &val_refs).unwrap();

        assert_eq!(cache.entry_count(TEST_LAYER), 3);

        // Then: individual push of 1 more entry
        let extra_key = pseudo_random_vec(TEST_DIM, TEST_KEY_SEED_3);
        let extra_val = pseudo_random_vec(TEST_DIM, TEST_VALUE_SEED_3);
        cache.push(TEST_LAYER, &extra_key, &extra_val).unwrap();

        assert_eq!(cache.entry_count(TEST_LAYER), 4);

        // Dequantize should give 4 entries
        let all_keys = cache.dequantize_all_keys(TEST_LAYER).unwrap();
        assert_eq!(all_keys.len(), 4);
    }

    // -- GPU export/import helper tests --------------------------------------

    #[test]
    fn collect_packed_data_empty() {
        let (bytes, scales) = collect_packed_data(&[]);
        assert!(bytes.is_empty());
        assert!(scales.is_empty());
    }

    #[test]
    fn collect_packed_data_roundtrip() {
        let config = test_config();
        let mut cache = QuantizedKVCache::new(config, TEST_NUM_LAYERS, TEST_QJL_SEED);
        let key = pseudo_random_vec(TEST_DIM, TEST_KEY_SEED);
        let val = pseudo_random_vec(TEST_DIM, TEST_VALUE_SEED);
        cache.push(TEST_LAYER, &key, &val).unwrap();

        // Export via collect_packed_data (tested function)
        let (packed, scales) = cache.export_packed_range(TEST_LAYER, 0, 1, true).unwrap();
        assert!(!packed.is_empty());
        assert_eq!(scales.len(), 1);

        // The exported data should allow reconstruction
        let key2 = pseudo_random_vec(TEST_DIM, TEST_KEY_SEED_2);
        let val2 = pseudo_random_vec(TEST_DIM, TEST_VALUE_SEED_2);
        cache.push(TEST_LAYER, &key2, &val2).unwrap();
        let (packed2, scales2) = cache.export_packed_range(TEST_LAYER, 0, 2, true).unwrap();
        assert_eq!(scales2.len(), 2);
        // First block's packed data should match
        let bytes_per_block = packed.len();
        assert_eq!(&packed2[..bytes_per_block], &packed[..]);
    }

    #[test]
    fn reconstruct_block_preserves_data() {
        let config = test_config();
        let mut cache = QuantizedKVCache::new(config, TEST_NUM_LAYERS, TEST_QJL_SEED);
        let key = pseudo_random_vec(TEST_DIM, TEST_KEY_SEED);
        let val = pseudo_random_vec(TEST_DIM, TEST_VALUE_SEED);
        cache.push(TEST_LAYER, &key, &val).unwrap();

        // Export, then reconstruct via import
        let (packed, scales) = cache.export_packed_range(TEST_LAYER, 0, 1, true).unwrap();
        let bytes_per_block = packed.len();
        const BITS_PER_BYTE: usize = 8;
        let signs_per_block = TEST_DIM.div_ceil(BITS_PER_BYTE);
        // Need QJL signs too — create dummy ones for the test
        let qjl_signs = vec![0u8; signs_per_block];
        let residual_norms = vec![0u16; 1];

        let import = PackedImport {
            layer: TEST_LAYER,
            polar_bits: BITS_3 - 1,
            packed_bytes: &packed,
            scales: &scales,
            qjl_signs_flat: &qjl_signs,
            residual_norms: &residual_norms,
            bytes_per_block,
            signs_per_block,
            is_keys: false,
        };
        let block = reconstruct_block(&import, 0);
        assert_eq!(block.polar_block().packed_indices(), &packed[..]);
        assert_eq!(block.polar_block().scale().to_bits(), scales[0]);
    }

    #[test]
    fn select_blocks_returns_correct_side() {
        let config = test_config();
        let mut cache = QuantizedKVCache::new(config, TEST_NUM_LAYERS, TEST_QJL_SEED);
        let key = pseudo_random_vec(TEST_DIM, TEST_KEY_SEED);
        let val = pseudo_random_vec(TEST_DIM, TEST_VALUE_SEED);
        cache.push(TEST_LAYER, &key, &val).unwrap();

        let keys = cache.select_blocks(TEST_LAYER, true);
        let vals = cache.select_blocks(TEST_LAYER, false);
        assert_eq!(keys.len(), 1);
        assert_eq!(vals.len(), 1);

        // Keys and values should have different packed data (different input vectors)
        assert_ne!(
            keys[0].polar_block().packed_indices(),
            vals[0].polar_block().packed_indices()
        );
    }

    #[test]
    fn select_blocks_mut_allows_push() {
        let config = test_config();
        let mut cache = QuantizedKVCache::new(config, TEST_NUM_LAYERS, TEST_QJL_SEED);
        assert_eq!(cache.select_blocks(TEST_LAYER, true).len(), 0);

        // Push via the normal API
        let key = pseudo_random_vec(TEST_DIM, TEST_KEY_SEED);
        let val = pseudo_random_vec(TEST_DIM, TEST_VALUE_SEED);
        cache.push(TEST_LAYER, &key, &val).unwrap();

        // select_blocks_mut should see the entry
        let keys_mut = cache.select_blocks_mut(TEST_LAYER, true);
        assert_eq!(keys_mut.len(), 1);
    }
}
