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

pub use config::{CacheConfig, QuantNormMode, QUANT_BLOCK_SIZE};
pub use pqo::PqoCache;
pub use precomputed::GpuPrecomputed;
pub use storage::{LayerBuffers, LayerStorage, QuantizedKV, StorageMetadata};
pub use tq::TqCache;

/// Helper: create a candle error from a string message.
pub(crate) fn cache_err(msg: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(format!("TurboQuant cache: {msg}"))
}

/// Lazy-initialize the shared `GpuPrecomputed` for a cache. Thread-safe:
/// concurrent callers may race, but only one result is stored (the loser's
/// result is dropped — deterministic so both are equivalent).
///
/// Internal helper. The `#[doc(hidden)] pub` visibility is a Rust convention
/// for items that are reachable from integration tests but not part of the
/// public API — no SemVer guarantees.
#[doc(hidden)]
// qual:allow(TQ-003) — rustqual false-positive (see docs/rustqual-bugs.md).
// Directly tested by `tests/cache_internals_tests.rs::ensure_gpu_precomputed`
// and `ensure_gpu_precomputed_returns_initialized_cell`, but rustqual's
// TQ_UNTESTED heuristic does not detect cross-crate integration tests even
// when the test name matches the function name exactly.
pub fn ensure_gpu_precomputed<'a>(
    cell: &'a OnceLock<GpuPrecomputed>,
    config: &CacheConfig,
    device: &Device,
) -> Result<&'a GpuPrecomputed> {
    if let Some(p) = cell.get() {
        return Ok(p);
    }
    let fresh = GpuPrecomputed::new(config, device)?;
    let _ = cell.set(fresh);
    match cell.get() {
        Some(p) => Ok(p),
        None => Err(cache_err(
            "precomputed cell unset after init — concurrent modification",
        )),
    }
}
