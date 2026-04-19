//! Post-launch CUDA error checking.
//!
//! CUDA kernel launches fail silently: `<<<grid, block>>>` returns void
//! and the only way to detect an invalid configuration, OOM, or shared
//! memory overflow is to query the runtime afterwards. Without this
//! check, a fused-attention kernel launched with bad parameters produces
//! garbage output that looks plausible.

#![cfg(feature = "cuda")]

use candle_core::{Error, Result};

use super::ffi;

/// Reads and clears the current CUDA error flag. Returns `Err` if the
/// previous launch failed, otherwise `Ok(())`.
// qual:allow(TQ-003) — CUDA-only, exercised via the `test_support` shim
// in `tests/cuda_error_check_tests.rs`; rustqual cannot see cross-feature
// integration tests.
pub(crate) fn check_cuda_kernel_launch() -> Result<()> {
    unsafe {
        let code = ffi::cudaGetLastError();
        if code == 0 {
            return Ok(());
        }
        let ptr = ffi::cudaGetErrorString(code);
        let msg = if ptr.is_null() {
            format!("CUDA kernel launch failed: code={}", code)
        } else {
            let cstr = std::ffi::CStr::from_ptr(ptr);
            format!(
                "CUDA kernel launch failed: {} (code={})",
                cstr.to_string_lossy(),
                code,
            )
        };
        Err(Error::msg(msg))
    }
}
