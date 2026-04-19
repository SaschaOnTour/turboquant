//! TurboQuant -- KV-Cache quantization with zero accuracy loss.
//!
//! Implements Google's TurboQuant algorithm (Zandieh et al., ICLR 2026)
//! for compressing LLM key-value caches to 3-4 bits per value.

pub mod attention;
pub mod codebook;
pub mod error;
pub mod math;
pub mod packed;
pub mod qjl;
pub mod quantize;
pub mod rotation;

#[cfg(feature = "candle")]
pub mod cache;

/// Test helpers shared by integration tests and benches. Declared `pub mod`
/// so cross-file test code can import them, and `#[doc(hidden)]` to keep
/// them out of rustdoc — but note that this module *is* part of the crate's
/// public API surface for SemVer purposes.
#[doc(hidden)]
pub mod test_utils;

pub use attention::{PackedImport, QuantizedKVCache};
pub use error::{Result, TurboQuantError};
pub use packed::PackedBlock;
pub use packed::TurboQuantConfig;
pub use qjl::{
    compute_qjl_signs, estimate_inner_product, estimate_inner_product_single,
    estimate_inner_product_with_codebook, precompute_query_projections, quantize_with_qjl,
    quantize_with_qjl_resources, EstimationContext, QjlBatchResources, QjlBlock,
};
pub use quantize::{
    dequantize_into_with_codebook, dequantize_vec, dequantize_vec_with_codebook, quantize_vec,
    quantize_vec_with_codebook,
};
pub use rotation::RotationOrder;
