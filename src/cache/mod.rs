//! Compressed KV-cache implementations for LLM inference.
//!
//! This module provides [`PqoCache`] (PolarQuant, covering both PQ and PQO
//! modes via `CacheConfig::outlier_blocks`) and [`TqCache`] (TurboQuant with
//! QJL correction).
//!
//! All implementations use the [`CompressedKVCache`] trait from
//! `mistralrs-kv-cache`, making them drop-in replacements for any
//! inference engine that supports the trait.
//!
//! Requires the `candle` feature flag.

mod common;
pub mod config;
pub mod cuda;
mod pqo;
mod precomputed;
pub(crate) mod quantize_tensor;
mod storage;
mod tq;
mod wht_tensor;

use std::sync::OnceLock;

use candle_core::{Device, Result};
use parking_lot::Mutex;

/// Lazy-initialization state for the shared [`GpuPrecomputed`] tensors.
///
/// Bundles the `OnceLock` (holding the initialized value) with a small
/// init mutex (serializing the slow path to avoid duplicate GPU
/// allocations). `PqoCache` and `TqCache` each own one of these.
///
/// Internal type exposed via `#[doc(hidden)] pub` so integration tests
/// can construct one for helpers like [`ensure_gpu_precomputed`]. Not
/// part of the public API.
#[doc(hidden)]
#[derive(Default)]
pub struct PrecomputedState {
    pub(crate) cell: OnceLock<GpuPrecomputed>,
    pub(crate) init_mutex: Mutex<()>,
}

pub use config::{CacheConfig, QuantNormMode, QUANT_BLOCK_SIZE};
pub use pqo::PqoCache;
pub use precomputed::GpuPrecomputed;
pub use storage::{LayerBuffers, LayerStorage, QuantizedKV, StorageMetadata};
pub use tq::TqCache;

/// Helper: create a candle error from a string message.
pub(crate) fn cache_err(msg: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(format!("TurboQuant cache: {msg}"))
}

/// Lazy-initialize the shared `GpuPrecomputed` for a cache.
///
/// Thread-safe via double-checked locking: the `init_mutex` serializes the
/// slow path so `GpuPrecomputed::new` runs at most once per cache instance,
/// even under contention. Subsequent callers take the fast path (a single
/// `OnceLock::get`) without touching the mutex.
///
/// The stable-Rust alternative `OnceLock::get_or_try_init` is still
/// nightly-only as of 1.95 (feature `once_cell_try`).
///
/// Internal helper. The `#[doc(hidden)] pub` visibility is a Rust convention
/// for items that are reachable from integration tests but not part of the
/// public API — no SemVer guarantees.
#[doc(hidden)]
// qual:allow(TQ-003) — rustqual false-positive; rationale below.
// Directly tested by `tests/cache_internals_tests.rs::ensure_gpu_precomputed`
// and `ensure_gpu_precomputed_returns_initialized_cell`, but rustqual's
// TQ_UNTESTED heuristic does not detect cross-crate integration tests even
// when the test name matches the function name exactly.
pub fn ensure_gpu_precomputed<'a>(
    state: &'a PrecomputedState,
    config: &CacheConfig,
    device: &Device,
) -> Result<&'a GpuPrecomputed> {
    if let Some(p) = state.cell.get() {
        return Ok(p);
    }
    // Slow path: serialize initialization to avoid wasted GPU allocations
    // when multiple threads race on the first prefill/decode call.
    let _init_guard = state.init_mutex.lock();
    if let Some(p) = state.cell.get() {
        return Ok(p);
    }
    let fresh = GpuPrecomputed::new(config, device)?;
    // `set` returns Err only if the cell was already populated; under the
    // init_mutex that should be impossible, so surface any such race
    // explicitly instead of silently discarding `fresh`.
    state
        .cell
        .set(fresh)
        .map_err(|_| cache_err("precomputed cell was initialized concurrently during set"))?;
    state
        .cell
        .get()
        .ok_or_else(|| cache_err("precomputed cell unset after successful set — unreachable"))
}
