//! Verifies that `check_cuda_kernel_launch` catches invalid kernel
//! launches instead of letting them pass silently.
//!
//! Regression test for issue #43. Requires both `cuda` and
//! `cuda-test-support` features: the latter compiles the deliberately
//! failing `tq_test_trigger_launch_error` kernel into the CUDA library.

#![cfg(all(feature = "cuda", feature = "cuda-test-support"))]

use candle_core::Device;
use turboquant::cache::cuda::check_cuda_kernel_launch;

// Links against the test-only CUDA helper compiled in the turboquant
// kernel library when the `cuda-test-support` feature is on. Declared
// here rather than in `src/` so no production source references a
// symbol that only exists in test builds.
extern "C" {
    fn tq_test_trigger_launch_error(stream: *const std::ffi::c_void);
}

fn cuda_device() -> Option<Device> {
    match Device::cuda_if_available(0) {
        Ok(dev) if dev.is_cuda() => Some(dev),
        _ => None,
    }
}

// qual:allow(complexity) — unsafe block is the FFI call into the
// intentionally-failing test kernel.
#[test]
fn check_cuda_kernel_launch_catches_invalid_launch() {
    let Some(device) = cuda_device() else {
        eprintln!("skipping: no CUDA device available");
        return;
    };
    let Device::Cuda(dev) = &device else {
        unreachable!("cuda_device returned non-CUDA device")
    };
    let stream = dev.cuda_stream().cu_stream() as *const std::ffi::c_void;
    unsafe { tq_test_trigger_launch_error(stream) };

    let err = check_cuda_kernel_launch().expect_err("invalid kernel launch must surface as Err");
    let msg = err.to_string();
    assert!(
        msg.contains("CUDA kernel launch failed"),
        "unexpected error message: {msg}",
    );
}

#[test]
fn check_cuda_kernel_launch_ok_on_clean_state() {
    if cuda_device().is_none() {
        eprintln!("skipping: no CUDA device available");
        return;
    }
    // Drain any leftover error state from prior tests in this process.
    let _ = check_cuda_kernel_launch();

    check_cuda_kernel_launch().expect("clean state must not produce a CUDA error");
}
