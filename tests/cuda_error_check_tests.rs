//! Verifies that `check_cuda_kernel_launch` catches invalid kernel
//! launches instead of letting them pass silently.
//!
//! Regression test for issue #43.

#![cfg(feature = "cuda")]

use candle_core::Device;

// The helper and the test trigger are both `pub(crate)` / not re-exported
// from the library, so the test reaches them through a tiny shim exposed
// in `turboquant::cache::cuda::test_support`.
use turboquant::cache::cuda::test_support::{
    check_cuda_kernel_launch_for_test, trigger_cuda_launch_error_for_test,
};

#[test]
fn check_cuda_kernel_launch_catches_invalid_launch() {
    let device = match Device::cuda_if_available(0) {
        Ok(dev) if dev.is_cuda() => dev,
        _ => {
            eprintln!("skipping: no CUDA device available");
            return;
        }
    };

    trigger_cuda_launch_error_for_test(&device).expect("helper launch should dispatch");
    let result = check_cuda_kernel_launch_for_test();

    assert!(
        result.is_err(),
        "check_cuda_kernel_launch must return Err after an invalid kernel launch",
    );
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("CUDA kernel launch failed"),
        "unexpected error message: {msg}",
    );
}

#[test]
fn check_cuda_kernel_launch_ok_on_clean_state() {
    let device = match Device::cuda_if_available(0) {
        Ok(dev) if dev.is_cuda() => dev,
        _ => {
            eprintln!("skipping: no CUDA device available");
            return;
        }
    };
    // Drain any leftover error state from prior tests in this process.
    let _ = check_cuda_kernel_launch_for_test();
    let _ = &device;

    check_cuda_kernel_launch_for_test().expect("clean state must not produce a CUDA error");
}
