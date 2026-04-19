//! CUDA kernel wrappers for TurboQuant operations.
//!
//! All modules are gated behind `#[cfg(feature = "cuda")]`.

#[cfg(feature = "cuda")]
pub mod attention;
#[cfg(feature = "cuda")]
pub(crate) mod ffi;
#[cfg(feature = "cuda")]
pub(crate) mod quantize;

#[cfg(feature = "cuda")]
mod error_check;

/// Post-launch CUDA error check. `#[doc(hidden)] pub` so the
/// `tests/cuda_error_check_tests.rs` integration test can call it from
/// the test crate; same SemVer-visible convention as
/// [`crate::test_utils`].
#[cfg(feature = "cuda")]
#[doc(hidden)]
pub use error_check::check_cuda_kernel_launch;
