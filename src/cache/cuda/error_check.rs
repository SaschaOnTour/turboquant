//! Post-launch CUDA error checking.
//!
//! CUDA kernel launches fail silently: `<<<grid, block>>>` returns void
//! and the only way to detect an invalid configuration, OOM, or shared
//! memory overflow is to query the runtime afterwards. Without this
//! check, a fused-attention kernel launched with bad parameters produces
//! garbage output that looks plausible.

#![cfg(feature = "cuda")]

use std::ffi::CStr;

use candle_core::{Error, Result};

use super::ffi;

/// Reads and clears the current CUDA error flag. Returns `Err` if the
/// previous launch failed, otherwise `Ok(())`.
// qual:allow(complexity) — both `unsafe` blocks are bare CUDA runtime
// calls (`cudaGetLastError`, `cudaGetErrorString`, `CStr::from_ptr`);
// splitting further would only shuffle FFI boilerplate.
pub fn check_cuda_kernel_launch() -> Result<()> {
    let code = unsafe { ffi::cudaGetLastError() };
    if code == 0 {
        return Ok(());
    }
    let msg_ptr = unsafe { ffi::cudaGetErrorString(code) };
    let msg = if msg_ptr.is_null() {
        format!("CUDA kernel launch failed: code={code}")
    } else {
        let cstr = unsafe { CStr::from_ptr(msg_ptr) };
        format!(
            "CUDA kernel launch failed: {} (code={code})",
            cstr.to_string_lossy(),
        )
    };
    Err(Error::msg(msg))
}
