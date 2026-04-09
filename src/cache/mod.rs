//! Compressed KV-cache implementations for LLM inference.
//!
//! This module provides [`PqoCache`] (PolarQuant Outlier) as the primary
//! cache implementation, with [`TqCache`] (TurboQuant with QJL correction)
//! and [`PqCache`] (plain PolarQuant) as alternatives.
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

pub use config::{CacheConfig, QuantNormMode, QUANT_BLOCK_SIZE};
pub use pqo::PqoCache;
pub use precomputed::GpuPrecomputed;
pub use storage::{CompressedStorage, QuantizedKV};
pub use tq::TqCache;

/// Helper: create a candle error from a string message.
pub(crate) fn cache_err(msg: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(format!("TurboQuant cache: {msg}"))
}
