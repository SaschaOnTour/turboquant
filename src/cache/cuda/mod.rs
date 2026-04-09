//! CUDA kernel wrappers for TurboQuant operations.
//!
//! All modules are gated behind `#[cfg(feature = "cuda")]`.

#[cfg(feature = "cuda")]
pub mod attention;
#[cfg(feature = "cuda")]
pub(crate) mod ffi;
#[cfg(feature = "cuda")]
pub(crate) mod quantize;
