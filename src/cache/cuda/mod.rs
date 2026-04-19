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

#[cfg(feature = "cuda")]
pub(crate) use error_check::check_cuda_kernel_launch;

/// Integration-test hooks for verifying the CUDA error-checking helper.
///
/// `#[doc(hidden)] pub` so `tests/cuda_error_check_tests.rs` can reach the
/// helpers without exposing them to downstream users. Not part of the
/// public API.
#[cfg(feature = "cuda")]
#[doc(hidden)]
pub mod test_support {
    use candle_core::{Device, Result};

    use super::ffi;

    // qual:allow(TQ-003) — integration-test shim, used from
    // `tests/cuda_error_check_tests.rs`.
    pub fn check_cuda_kernel_launch_for_test() -> Result<()> {
        super::check_cuda_kernel_launch()
    }

    /// Launches a no-op kernel with an invalid block dimension (>1024
    /// threads/block) so the CUDA runtime records
    /// `cudaErrorInvalidConfiguration` in the current thread's error
    /// state. The subsequent [`check_cuda_kernel_launch_for_test`] call
    /// should observe it.
    // qual:allow(TQ-003) — integration-test shim, used from
    // `tests/cuda_error_check_tests.rs`.
    pub fn trigger_cuda_launch_error_for_test(device: &Device) -> Result<()> {
        let Device::Cuda(dev) = device else {
            candle_core::bail!("trigger_cuda_launch_error_for_test requires a CUDA device");
        };
        let stream = dev.cuda_stream().cu_stream() as *const std::ffi::c_void;
        unsafe { ffi::tq_test_trigger_launch_error(stream) };
        Ok(())
    }
}
