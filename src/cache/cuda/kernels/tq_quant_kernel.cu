/*
 * tq_quant_kernel.cu — CUDA quantize kernel for TurboQuant.
 *
 * Two kernels:
 *   1. tq_quant_kernel  — PolarQuant: normalize → sign flip → WHT → quantize → pack
 *   2. tq_qjl_kernel    — QJL correction: compute residual, project through
 *                          Rademacher matrix, pack sign bits
 *
 * Each CUDA block processes one vector (one KV head × one token).
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <float.h>

#include "tq_common.h"

/* -----------------------------------------------------------------------
 * Warp-level reduction for sum of squares (L2 norm squared).
 * ----------------------------------------------------------------------- */

__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ float block_reduce_sum(float val, int tid, int block_size) {
    /* Reduce within each warp */
    val = warp_reduce_sum(val);

    /* Collect warp results in shared memory */
    __shared__ float warp_sums[32]; /* max 32 warps per block */
    int warp_id = tid >> 5;
    int lane_id = tid & 31;

    if (lane_id == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    /* First warp reduces the warp sums */
    int num_warps = (block_size + 31) >> 5;
    val = (tid < num_warps) ? warp_sums[tid] : 0.0f;
    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}

/* -----------------------------------------------------------------------
 * PolarQuant kernel (L2Norm mode — original)
 *
 * Grid:  num_blocks (one per vector)
 * Block: block_size threads (e.g. 32)
 *
 * Algorithm:
 *   1. Load input, compute L2 norm (parallel reduction)
 *   2. Normalize to unit vector
 *   3. Sign flip (forward rotation step 1)
 *   4. Forward WHT + normalize by 1/sqrt(dim)
 *   5. Nearest centroid via linear scan of boundaries
 *   6. Cooperative bit packing into output
 * ----------------------------------------------------------------------- */

__global__ void tq_quant_kernel(
    const float    *__restrict__ input,
    const float    *__restrict__ boundaries,
    uint64_t        sign_seed,
    uint8_t        *__restrict__ packed_out,
    uint16_t       *__restrict__ scales_out,
    int             head_dim,
    int             bits,
    int             num_boundaries,
    int             bytes_per_block)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    extern __shared__ char smem[];
    float   *s_data   = (float *)smem;
    uint8_t *s_packed = (uint8_t *)(smem + head_dim * sizeof(float));

    /* 1. Load input + compute L2 norm */
    float val = input[bid * head_dim + tid];
    float sq = val * val;
    float sum_sq = block_reduce_sum(sq, tid, head_dim);

    __shared__ float s_norm;
    if (tid == 0) {
        s_norm = sqrtf(sum_sq);
    }
    __syncthreads();

    float norm = s_norm;

    /* 2. Normalize */
    float normalized = (norm > 1e-10f) ? (val / norm) : 0.0f;

    /* 3. Sign flip (forward rotation) */
    float sign = tq_sign(sign_seed, tid);
    s_data[tid] = normalized * sign;
    __syncthreads();

    /* 4. Forward WHT */
    for (int h = 1; h < head_dim; h <<= 1) {
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

    /* Normalize WHT output */
    float rotated = s_data[tid] * rsqrtf((float)head_dim);

    /* 5. Nearest centroid (linear scan on boundaries) */
    uint8_t idx = 0;
    for (int b = 0; b < num_boundaries; b++) {
        if (rotated > boundaries[b]) {
            idx = (uint8_t)(b + 1);
        }
    }

    /* 6. Pack indices */
    /* Zero the packed output first (cooperative) */
    if (tid < bytes_per_block) {
        s_packed[tid] = 0;
    }
    __syncthreads();

    /* Each thread atomically ORs its index bits into the correct byte */
    if (bits == 2) {
        int byte_idx = tid >> 2;
        int shift = (tid & 3) << 1;
        atomicOr((unsigned int *)(s_packed + (byte_idx & ~3)),
                 (unsigned int)((idx & 0x3) << shift) << ((byte_idx & 3) * 8));
    } else if (bits == 3) {
        /* 3-bit packing: 8 indices per 3 bytes — use atomicOr on bytes */
        int group = tid >> 3;
        int pos   = tid & 7;
        int base  = group * 3;
        /* Each position writes to specific bit positions.
         * We use atomicOr on uint8 via atomicOr on aligned uint32. */
        uint32_t b0_bits = 0, b1_bits = 0, b2_bits = 0;
        switch (pos) {
            case 0: b0_bits = (idx & 0x7);        break;
            case 1: b0_bits = (idx & 0x7) << 3;   break;
            case 2: b0_bits = (idx & 0x3) << 6;
                    b1_bits = (idx >> 2)  & 0x1;   break;
            case 3: b1_bits = (idx & 0x7) << 1;   break;
            case 4: b1_bits = (idx & 0x7) << 4;   break;
            case 5: b1_bits = (idx & 0x1) << 7;
                    b2_bits = (idx >> 1)  & 0x3;   break;
            case 6: b2_bits = (idx & 0x7) << 2;   break;
            case 7: b2_bits = (idx & 0x7) << 5;   break;
        }
        if (b0_bits) atomicOr((unsigned int *)(s_packed + (base & ~3)),
                              b0_bits << ((base & 3) * 8));
        if (b1_bits) atomicOr((unsigned int *)(s_packed + ((base + 1) & ~3)),
                              b1_bits << (((base + 1) & 3) * 8));
        if (b2_bits) atomicOr((unsigned int *)(s_packed + ((base + 2) & ~3)),
                              b2_bits << (((base + 2) & 3) * 8));
    } else {
        /* 4-bit packing */
        int byte_idx = tid >> 1;
        int shift = (tid & 1) << 2;
        atomicOr((unsigned int *)(s_packed + (byte_idx & ~3)),
                 (unsigned int)((idx & 0xF) << shift) << ((byte_idx & 3) * 8));
    }
    __syncthreads();

    /* Write packed output */
    if (tid < bytes_per_block) {
        packed_out[bid * bytes_per_block + tid] = s_packed[tid];
    }

    /* Write scale (f16) */
    if (tid == 0) {
        scales_out[bid] = __half_as_ushort(__float2half(norm));
    }
}

/* -----------------------------------------------------------------------
 * Warp-level max reduction
 * ----------------------------------------------------------------------- */

__device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ float block_reduce_max(float val, int tid, int block_size) {
    val = warp_reduce_max(val);
    __shared__ float warp_maxes[32];
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    if (lane_id == 0) warp_maxes[warp_id] = val;
    __syncthreads();
    int num_warps = (block_size + 31) >> 5;
    val = (tid < num_warps) ? warp_maxes[tid] : -FLT_MAX;
    if (warp_id == 0) val = warp_reduce_max(val);
    return val;
}

/* -----------------------------------------------------------------------
 * PolarQuant kernel — MaxNorm mode (llama.cpp compatible)
 *
 * Grid:  num_blocks (one per 32-element block)
 * Block: block_size threads (32)
 *
 * Algorithm (matches Rust polar_quantize MaxNorm):
 *   1. Load input
 *   2. Forward WHT (butterfly) + 1/sqrt(dim)
 *   3. Sign flip AFTER WHT (signs[tid] * WHT(x)[tid])
 *   4. Find max absolute value → scale = amax / outer_centroid
 *   5. Divide by scale
 *   6. Nearest centroid via boundaries
 *   7. Cooperative bit packing
 *   8. Store scale * scale_sign as F16
 * ----------------------------------------------------------------------- */

/* -----------------------------------------------------------------------
 * MaxNorm quantize-and-pack kernel
 *
 * Input: ALREADY rotated values (WHT + signs applied by caller).
 * This kernel only does: amax → scale → divide → quantize → pack.
 * WHT is done externally (Candle butterfly_wht_forward_gpu) to ensure
 * exact numerical consistency with the CUDA dequant butterfly.
 * ----------------------------------------------------------------------- */
__global__ void tq_quant_maxnorm_kernel(
    const float    *__restrict__ rotated_input,
    const float    *__restrict__ boundaries,
    uint8_t        *__restrict__ packed_out,
    uint16_t       *__restrict__ scales_out,
    int             block_size,
    int             bits,
    int             num_boundaries,
    int             bytes_per_block,
    float           outer_centroid,
    float           scale_sign)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    extern __shared__ char smem[];
    uint8_t *s_packed = (uint8_t *)smem;

    /* 1. Load already-rotated value */
    float rotated = rotated_input[bid * block_size + tid];

    /* 2. MaxNorm: find max abs → scale (warp-level for block_size=32) */
    float abs_val = fabsf(rotated);
    float amax = warp_reduce_max(abs_val);

    float raw_scale = amax / outer_centroid;
    float scale = (raw_scale > 1e-10f) ? raw_scale : 1e-10f;

    /* 3. Divide by scale */
    float scaled = rotated / scale;

    /* 6. Nearest centroid (linear scan on boundaries) */
    uint8_t idx = 0;
    for (int b = 0; b < num_boundaries; b++) {
        if (scaled > boundaries[b]) {
            idx = (uint8_t)(b + 1);
        }
    }

    /* 7. Pack indices */
    if (tid < bytes_per_block) {
        s_packed[tid] = 0;
    }
    __syncthreads();

    if (bits == 2) {
        int byte_idx = tid >> 2;
        int shift = (tid & 3) << 1;
        atomicOr((unsigned int *)(s_packed + (byte_idx & ~3)),
                 (unsigned int)((idx & 0x3) << shift) << ((byte_idx & 3) * 8));
    } else if (bits == 3) {
        int group = tid >> 3;
        int pos   = tid & 7;
        int base  = group * 3;
        uint32_t b0_bits = 0, b1_bits = 0, b2_bits = 0;
        switch (pos) {
            case 0: b0_bits = (idx & 0x7);        break;
            case 1: b0_bits = (idx & 0x7) << 3;   break;
            case 2: b0_bits = (idx & 0x3) << 6;
                    b1_bits = (idx >> 2)  & 0x1;   break;
            case 3: b1_bits = (idx & 0x7) << 1;   break;
            case 4: b1_bits = (idx & 0x7) << 4;   break;
            case 5: b1_bits = (idx & 0x1) << 7;
                    b2_bits = (idx >> 1)  & 0x3;   break;
            case 6: b2_bits = (idx & 0x7) << 2;   break;
            case 7: b2_bits = (idx & 0x7) << 5;   break;
        }
        if (b0_bits) atomicOr((unsigned int *)(s_packed + (base & ~3)),
                              b0_bits << ((base & 3) * 8));
        if (b1_bits) atomicOr((unsigned int *)(s_packed + ((base + 1) & ~3)),
                              b1_bits << (((base + 1) & 3) * 8));
        if (b2_bits) atomicOr((unsigned int *)(s_packed + ((base + 2) & ~3)),
                              b2_bits << (((base + 2) & 3) * 8));
    } else {
        int byte_idx = tid >> 1;
        int shift = (tid & 1) << 2;
        atomicOr((unsigned int *)(s_packed + (byte_idx & ~3)),
                 (unsigned int)((idx & 0xF) << shift) << ((byte_idx & 3) * 8));
    }
    __syncthreads();

    if (tid < bytes_per_block) {
        packed_out[bid * bytes_per_block + tid] = s_packed[tid];
    }

    /* 8. Store scale as F16, with sign (negative = outlier codebook) */
    if (tid == 0) {
        scales_out[bid] = __half_as_ushort(__float2half(scale * scale_sign));
    }
}

/* -----------------------------------------------------------------------
 * QJL sign kernel
 *
 * Grid:  num_blocks (one per vector)
 * Block: head_dim threads
 *
 * After PolarQuant, computes the residual and generates QJL sign bits.
 *
 * Parameters:
 *   original    — [num_blocks * head_dim] original input vectors
 *   dequantized — [num_blocks * head_dim] dequantized polar vectors
 *   qjl_seed    — seed for Rademacher matrix
 *   qjl_signs_out    — [num_blocks * signs_per_block] packed sign bits
 *   residual_norms_out — [num_blocks] residual L2 norms as uint16 (f16)
 *   head_dim    — vector dimension
 *   signs_per_block — ceil(head_dim / 8)
 * ----------------------------------------------------------------------- */

__global__ void tq_qjl_kernel(
    const float    *__restrict__ original,
    const float    *__restrict__ dequantized,
    uint64_t        qjl_seed,
    uint8_t        *__restrict__ qjl_signs_out,
    uint16_t       *__restrict__ residual_norms_out,
    int             head_dim,
    int             signs_per_block)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    /* 1. Compute residual */
    float residual = original[bid * head_dim + tid]
                   - dequantized[bid * head_dim + tid];

    /* 2. Compute residual L2 norm */
    float sq = residual * residual;
    float sum_sq = block_reduce_sum(sq, tid, head_dim);

    __shared__ float s_residual_norm;
    if (tid == 0) {
        s_residual_norm = sqrtf(sum_sq);
    }
    __syncthreads();

    /* 3. Compute Rademacher projection: sum_col(R[tid][col] * residual[col])
     * R[row][col] = rademacher_sign(seed, row, col) / sqrt(dim)
     *
     * Each thread computes one row of the projection (its own sign bit).
     * This requires reading ALL residual values — we use shared memory. */
    extern __shared__ float s_residual[];
    s_residual[tid] = residual;
    __syncthreads();

    float projection = 0.0f;
    float inv_sqrt_dim = rsqrtf((float)head_dim);
    for (int col = 0; col < head_dim; col++) {
        float r_sign = tq_rademacher_sign(qjl_seed, tid, col);
        projection += r_sign * inv_sqrt_dim * s_residual[col];
    }

    /* 4. Pack sign bit: 1 if projection >= 0, 0 otherwise */
    uint8_t sign_bit = (projection >= 0.0f) ? 1 : 0;

    /* Cooperative byte packing via atomicOr */
    __shared__ uint8_t s_signs[128]; /* max signs_per_block */
    if (tid < signs_per_block) {
        s_signs[tid] = 0;
    }
    __syncthreads();

    int byte_idx = tid >> 3;
    int bit_pos  = tid & 7;
    atomicOr((unsigned int *)(s_signs + (byte_idx & ~3)),
             (unsigned int)(sign_bit << bit_pos) << ((byte_idx & 3) * 8));
    __syncthreads();

    /* Write output */
    if (tid < signs_per_block) {
        qjl_signs_out[bid * signs_per_block + tid] = s_signs[tid];
    }
    if (tid == 0) {
        residual_norms_out[bid] = __half_as_ushort(__float2half(s_residual_norm));
    }
}

/* -----------------------------------------------------------------------
 * C-linkage launchers
 * ----------------------------------------------------------------------- */

extern "C" void tq_quant_batch(
    const float    *input,
    const float    *boundaries,
    uint64_t        sign_seed,
    uint8_t        *packed_out,
    uint16_t       *scales_out,
    int32_t         num_blocks,
    int32_t         head_dim,
    int32_t         bits,
    int32_t         num_boundaries,
    int32_t         bytes_per_block,
    cudaStream_t    stream)
{
    if (num_blocks == 0) return;

    /* Shared: head_dim floats + bytes_per_block packed (aligned to 4) */
    int packed_aligned = bytes_per_block + ((bytes_per_block & 3) ? (4 - (bytes_per_block & 3)) : 0);
    size_t smem_size = head_dim * sizeof(float) + packed_aligned;

    tq_quant_kernel<<<num_blocks, head_dim, smem_size, stream>>>(
        input, boundaries, sign_seed,
        packed_out, scales_out,
        head_dim, bits, num_boundaries, bytes_per_block);
}

/* -----------------------------------------------------------------------
 * Standalone bit-packing kernel
 *
 * Grid:  num_vectors (one per vector of block_size elements)
 * Block: block_size threads (e.g. 32)
 *
 * Takes U8 indices [num_vectors, block_size] and packs them to
 * [num_vectors, bytes_per_block] according to the bit width.
 * ----------------------------------------------------------------------- */
__global__ void tq_pack_kernel(
    const uint8_t  *__restrict__ indices,
    uint8_t        *__restrict__ packed_out,
    int             block_size,
    int             bits,
    int             bytes_per_block)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    extern __shared__ uint8_t s_packed[];

    uint8_t idx = indices[bid * block_size + tid];

    if (tid < bytes_per_block) {
        s_packed[tid] = 0;
    }
    __syncthreads();

    if (bits == 2) {
        int byte_idx = tid >> 2;
        int shift = (tid & 3) << 1;
        atomicOr((unsigned int *)(s_packed + (byte_idx & ~3)),
                 (unsigned int)((idx & 0x3) << shift) << ((byte_idx & 3) * 8));
    } else if (bits == 3) {
        int group = tid >> 3;
        int pos   = tid & 7;
        int base  = group * 3;
        uint32_t b0_bits = 0, b1_bits = 0, b2_bits = 0;
        switch (pos) {
            case 0: b0_bits = (idx & 0x7);        break;
            case 1: b0_bits = (idx & 0x7) << 3;   break;
            case 2: b0_bits = (idx & 0x3) << 6;
                    b1_bits = (idx >> 2)  & 0x1;   break;
            case 3: b1_bits = (idx & 0x7) << 1;   break;
            case 4: b1_bits = (idx & 0x7) << 4;   break;
            case 5: b1_bits = (idx & 0x1) << 7;
                    b2_bits = (idx >> 1)  & 0x3;   break;
            case 6: b2_bits = (idx & 0x7) << 2;   break;
            case 7: b2_bits = (idx & 0x7) << 5;   break;
        }
        if (b0_bits) atomicOr((unsigned int *)(s_packed + (base & ~3)),
                              b0_bits << ((base & 3) * 8));
        if (b1_bits) atomicOr((unsigned int *)(s_packed + ((base + 1) & ~3)),
                              b1_bits << (((base + 1) & 3) * 8));
        if (b2_bits) atomicOr((unsigned int *)(s_packed + ((base + 2) & ~3)),
                              b2_bits << (((base + 2) & 3) * 8));
    } else {
        int byte_idx = tid >> 1;
        int shift = (tid & 1) << 2;
        atomicOr((unsigned int *)(s_packed + (byte_idx & ~3)),
                 (unsigned int)((idx & 0xF) << shift) << ((byte_idx & 3) * 8));
    }
    __syncthreads();

    if (tid < bytes_per_block) {
        packed_out[bid * bytes_per_block + tid] = s_packed[tid];
    }
}

extern "C" void tq_quant_maxnorm_batch(
    const float    *rotated_input,
    const float    *boundaries,
    uint8_t        *packed_out,
    uint16_t       *scales_out,
    int32_t         num_blocks,
    int32_t         block_size,
    int32_t         bits,
    int32_t         num_boundaries,
    int32_t         bytes_per_block,
    float           outer_centroid,
    float           scale_sign,
    cudaStream_t    stream)
{
    if (num_blocks == 0) return;

    int packed_aligned = bytes_per_block + ((bytes_per_block & 3) ? (4 - (bytes_per_block & 3)) : 0);

    tq_quant_maxnorm_kernel<<<num_blocks, block_size, packed_aligned, stream>>>(
        rotated_input, boundaries,
        packed_out, scales_out,
        block_size, bits, num_boundaries, bytes_per_block,
        outer_centroid, scale_sign);
}

extern "C" void tq_pack_indices(
    const uint8_t  *indices,
    uint8_t        *packed_out,
    int32_t         num_vectors,
    int32_t         block_size,
    int32_t         bits,
    int32_t         bytes_per_block,
    cudaStream_t    stream)
{
    if (num_vectors == 0) return;

    int packed_aligned = bytes_per_block + ((bytes_per_block & 3) ? (4 - (bytes_per_block & 3)) : 0);
    tq_pack_kernel<<<num_vectors, block_size, packed_aligned, stream>>>(
        indices, packed_out,
        block_size, bits, bytes_per_block);
}

extern "C" void tq_qjl_batch(
    const float    *original,
    const float    *dequantized,
    uint64_t        qjl_seed,
    uint8_t        *qjl_signs_out,
    uint16_t       *residual_norms_out,
    int32_t         num_blocks,
    int32_t         head_dim,
    int32_t         signs_per_block,
    cudaStream_t    stream)
{
    if (num_blocks == 0) return;

    /* Shared: head_dim floats for residual */
    size_t smem_size = head_dim * sizeof(float);

    tq_qjl_kernel<<<num_blocks, head_dim, smem_size, stream>>>(
        original, dequantized, qjl_seed,
        qjl_signs_out, residual_norms_out,
        head_dim, signs_per_block);
}
