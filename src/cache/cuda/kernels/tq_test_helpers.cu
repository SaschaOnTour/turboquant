/*
 * tq_test_helpers.cu — Test-only CUDA helpers.
 *
 * Launches that deliberately fail, used by Rust integration tests to
 * verify that the error-checking helpers actually catch CUDA kernel
 * launch failures instead of letting them pass silently.
 */

#include <cuda_runtime.h>

__global__ void tq_test_noop() {}

extern "C" void tq_test_trigger_launch_error(const void *stream_opaque) {
    cudaStream_t stream = (cudaStream_t)stream_opaque;
    // 2048 threads per block exceeds the 1024 hardware maximum on every
    // CUDA-capable GPU since compute capability 2.0, so the launch fails
    // with cudaErrorInvalidConfiguration.
    dim3 grid(1);
    dim3 block(2048);
    tq_test_noop<<<grid, block, 0, stream>>>();
}
