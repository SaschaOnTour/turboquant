/*
 * tq_dequant_kernel.cu — CUDA dequantize kernel for TurboQuant.
 *
 * Dequantizes packed polar blocks back to f32 vectors.
 * Each CUDA block processes one quantized block (block_size elements).
 *
 * Algorithm per block:
 *   1. Cooperatively load packed bytes into shared memory
 *   2. Each thread unpacks its own index
 *   3. Codebook lookup → shared memory
 *   4. Inverse WHT (butterfly stages with __syncthreads)
 *   5. Normalize by 1/sqrt(block_size), apply sign flip, scale by norm
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "tq_common.h"

/* -----------------------------------------------------------------------
 * Dequantize kernel
 *
 * Grid:  num_blocks  (one block per quantized block)
 * Block: block_size threads (one thread per dimension element)
 *
 * Parameters:
 *   packed_indices  — [num_blocks * bytes_per_block] packed index bytes
 *   scales          — [num_blocks] f16 scale factors
 *   codebook        — [num_centroids] centroid values as float
 *   sign_pattern    — [block_size] sign values (+1.0 or -1.0) as float
 *   output          — [num_blocks * block_size] output f32 values
 *   block_size      — quantization block size (32)
 *   bits            — polar block bit width (2, 3, or 4)
 *   bytes_per_block — packed bytes per block
 * ----------------------------------------------------------------------- */

__global__ void tq_dequant_kernel(
    const uint8_t  *__restrict__ packed_indices,
    const uint16_t *__restrict__ scales,
    const float    *__restrict__ codebook,
    const float    *__restrict__ sign_pattern,
    float          *__restrict__ output,
    int             block_size,
    int             bits,
    int             bytes_per_block)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    /* 1. Load packed bytes into shared memory */
    extern __shared__ char smem[];
    uint8_t *s_packed = (uint8_t *)smem;
    float   *s_data   = (float *)(smem + bytes_per_block +
                         /* align to 4 bytes */
                         ((bytes_per_block & 3) ? (4 - (bytes_per_block & 3)) : 0));

    const uint8_t *block_packed = packed_indices + bid * bytes_per_block;
    if (tid < bytes_per_block) {
        s_packed[tid] = block_packed[tid];
    }
    __syncthreads();

    /* 2. Unpack index for this thread's dimension */
    uint8_t idx;
    if (bits == 2) {
        idx = tq_unpack_2bit(s_packed, tid);
    } else if (bits == 3) {
        idx = tq_unpack_3bit(s_packed, tid);
    } else {
        idx = tq_unpack_4bit(s_packed, tid);
    }

    /* 3. Codebook lookup + sign flip BEFORE WHT */
    /* The inverse rotation is: H_norm @ (signs * codebook_values * scale) */
    /* So signs must be applied before the WHT butterfly, not after. */
    s_data[tid] = codebook[idx] * sign_pattern[tid];
    __syncthreads();

    /* 4. Inverse WHT: butterfly stages */
    for (int h = 1; h < block_size; h <<= 1) {
        int half_stride = h;
        int full_stride = h << 1;
        if ((tid % full_stride) < half_stride) {
            float x = s_data[tid];
            float y = s_data[tid + half_stride];
            s_data[tid]                = x + y;
            s_data[tid + half_stride]  = x - y;
        }
        __syncthreads();
    }

    /* 5. Normalize + scale (sign already applied in step 3) */
    float norm_factor = rsqrtf((float)block_size);
    float scale = __half2float(*((const __half *)&scales[bid]));

    output[bid * block_size + tid] = s_data[tid] * norm_factor * fabsf(scale);
}

/* -----------------------------------------------------------------------
 * C-linkage launcher (called from Rust FFI)
 * ----------------------------------------------------------------------- */

extern "C" void tq_dequant_batch(
    const uint8_t  *packed_indices,
    const uint16_t *scales,
    const float    *codebook,
    const float    *sign_pattern,
    float          *output,
    int32_t         num_blocks,
    int32_t         block_size,
    int32_t         bits,
    int32_t         bytes_per_block,
    cudaStream_t    stream)
{
    if (num_blocks == 0) return;

    /* Shared memory: packed bytes (aligned to 4) + block_size floats */
    int packed_aligned = bytes_per_block + ((bytes_per_block & 3) ? (4 - (bytes_per_block & 3)) : 0);
    size_t smem_size = packed_aligned + block_size * sizeof(float);

    tq_dequant_kernel<<<num_blocks, block_size, smem_size, stream>>>(
        packed_indices, scales, codebook, sign_pattern,
        output, block_size, bits, bytes_per_block);
}
