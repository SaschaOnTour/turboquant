/*
 * tq_attention_kernel.cu — Fused compressed-cache attention for TurboQuant.
 *
 * Key optimization: multiple warps cooperate on the SAME token.
 * Each warp dequants a different quant-block (qb) of the same token,
 * so all 4 qblocks of K (or V) are dequanted in parallel.
 *
 * For head_dim=128, num_qblocks=4:
 *   WARPS_PER_TOKEN = 4 (one warp per qblock)
 *   NUM_TOKEN_GROUPS = TQ_NUM_WARPS / 4
 *
 * Butterfly WHT via __shfl_xor_sync (register-only).
 * Codebook + sign in shared memory (loaded once).
 * Partitioned for GPU saturation.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <float.h>

#include "tq_common.h"

#define TQ_NUM_WARPS 8
#define TQ_NUM_THREADS (TQ_NUM_WARPS * 32)
#define TQ_MAX_QBLOCKS 8
#define TQ_PARTITION_SIZE 512
#define TQ_MAX_CODEBOOK 16

/* -----------------------------------------------------------------------
 * Register-based dequant via warp shuffle
 * ----------------------------------------------------------------------- */
__device__ __forceinline__
float dequant_register(
    uint8_t *s_packed,
    const uint8_t *packed_ptr, float scale,
    const float *s_codebook, const float *s_sign,
    int lane, int bits, int bytes_per_qblock)
{
    if (lane < bytes_per_qblock) s_packed[lane] = packed_ptr[lane];
    __syncwarp();
    uint8_t idx;
    if (bits == 2)      idx = tq_unpack_2bit(s_packed, lane);
    else if (bits == 3) idx = tq_unpack_3bit(s_packed, lane);
    else                idx = tq_unpack_4bit(s_packed, lane);
    float val = s_codebook[idx] * s_sign[lane]; /* sign BEFORE WHT */
    #pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
        float other = __shfl_xor_sync(0xffffffff, val, h);
        val = (lane & h) ? (other - val) : (val + other);
    }
    return val * 0.1767766953f * scale; /* no sign here */
}

/* -----------------------------------------------------------------------
 * Partitioned attention kernel — multi-warp-per-token
 *
 * WARPS_PER_TOKEN warps cooperate on the same token:
 *   warp_in_group handles qblock = warp_in_group
 *   All K qblocks dequanted in parallel → partial QK reduced across group
 *   All V qblocks dequanted in parallel → each warp accumulates its own qblock
 * ----------------------------------------------------------------------- */
__global__ void tq_fused_attention_decode(
    const float    *__restrict__ q,
    const uint8_t  *__restrict__ k_indices,
    const uint16_t *__restrict__ k_scales,
    const uint8_t  *__restrict__ v_indices,
    const uint16_t *__restrict__ v_scales,
    const float    *__restrict__ codebook,
    const float    *__restrict__ sign_pattern,
    float          *__restrict__ partial_out,
    float          *__restrict__ partial_max,
    float          *__restrict__ partial_sum,
    /* QJL correction (nullable: pass nullptr when qjl_enabled=0) */
    const uint8_t  *__restrict__ qjl_signs,
    const uint16_t *__restrict__ qjl_residual_norms,
    const float    *__restrict__ rq,  /* Rademacher-projected query [num_attention_heads, head_dim] */
    int             qjl_enabled,
    int             num_attention_heads,
    int             num_kv_heads,
    int             head_dim,
    int             kv_len,
    int             kv_stride,
    int             packed_dim,
    int             num_qblocks,
    int             bits,
    float           softmax_scale,
    int             num_partitions)
{
    const int head = blockIdx.x;
    const int part = blockIdx.y;
    const int num_queries_per_kv = num_attention_heads / num_kv_heads;
    const int kv_head = head / num_queries_per_kv;

    const int tid = threadIdx.x;
    const int warp_idx = tid / 32;
    const int lane = tid % 32;
    const int bytes_per_qblock = 32 * bits / 8;

    /* Warp grouping: WARPS_PER_TOKEN warps cooperate on one token */
    const int warps_per_token = num_qblocks;  /* 4 for head_dim=128 */
    const int num_token_groups = TQ_NUM_WARPS / warps_per_token;
    const int group_idx = warp_idx / warps_per_token;
    const int qb = warp_idx % warps_per_token;  /* which qblock this warp handles */

    const int part_start = part * TQ_PARTITION_SIZE;
    const int part_end = min(part_start + TQ_PARTITION_SIZE, kv_len);
    if (part_start >= kv_len) return;

    /* Shared memory */
    extern __shared__ char smem_raw[];
    float   *s_codebook = (float *)smem_raw;
    float   *s_sign     = s_codebook + TQ_MAX_CODEBOOK;
    uint8_t *s_packed   = (uint8_t *)(s_sign + 32);
    uint8_t *my_packed  = s_packed + warp_idx * 16;

    /* Per-group QK reduction: partial sums from each warp in the group */
    float *s_group_qk = (float *)(s_packed + TQ_NUM_WARPS * 16);
    /* s_group_qk layout: [num_token_groups][warps_per_token] */

    /* Warp reduction area */
    float *s_group_max = s_group_qk + TQ_NUM_WARPS;
    float *s_group_sum = s_group_max + num_token_groups;
    float *s_group_acc = s_group_sum + num_token_groups;
    /* s_group_acc: [num_token_groups][head_dim] — each group writes its full output */

    /* Load codebook + sign once */
    int num_centroids = (1 << bits);
    if (tid < num_centroids) s_codebook[tid] = codebook[tid];
    if (tid < 32) s_sign[tid] = sign_pattern[tid];
    __syncthreads();

    /* Load query for this warp's qblock */
    const float *q_head = q + head * head_dim;
    float q_val = q_head[qb * 32 + lane];

    /* QJL: load precomputed Rademacher-projected query r_q = R^T @ q.
     * Computed on host via matmul (fast), passed as kernel parameter. */
    float rq_val = 0.0f;
    if (qjl_enabled) {
        rq_val = rq[head * head_dim + qb * 32 + lane];
    }

    /* Per-group online softmax (all warps in group share same state via registers) */
    float group_max = -FLT_MAX;
    float group_sum = 0.0f;
    float out_acc = 0.0f;  /* this warp's qblock accumulator (32 elements in lane) */

    const uint8_t  *ki_base = k_indices + kv_head * kv_stride * packed_dim;
    const uint16_t *ks_base = k_scales  + kv_head * kv_stride * num_qblocks;
    const uint8_t  *vi_base = v_indices + kv_head * kv_stride * packed_dim;
    const uint16_t *vs_base = v_scales  + kv_head * kv_stride * num_qblocks;

    /* QJL base pointers (only valid when qjl_enabled) */
    const int signs_per_head = head_dim / 8;
    const uint8_t  *qjl_signs_base = qjl_enabled
        ? qjl_signs + kv_head * kv_stride * signs_per_head : nullptr;
    const uint16_t *qjl_norms_base = qjl_enabled
        ? qjl_residual_norms + kv_head * kv_stride : nullptr;

    for (int token = part_start + group_idx; token < part_end; token += num_token_groups) {
        /* --- K dequant: each warp dequants its qblock in parallel --- */
        const uint8_t *kp = ki_base + token * packed_dim + qb * bytes_per_qblock;
        float ks = fabsf(__half2float(
            *((const __half *)(ks_base + token * num_qblocks + qb))));

        float k_val = dequant_register(my_packed, kp, ks,
                                       s_codebook, s_sign,
                                       lane, bits, bytes_per_qblock);

        /* Partial QK: dot product for this qblock (32 elements) */
        float partial_qk = q_val * k_val;
        /* Warp-local reduce (sum 32 lanes) */
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            partial_qk += __shfl_xor_sync(0xffffffff, partial_qk, offset);

        /* Cross-warp reduce within group: sum partial QK from all qblocks */
        if (lane == 0)
            s_group_qk[group_idx * warps_per_token + qb] = partial_qk;
        __syncthreads();

        float score = 0.0f;
        if (lane == 0) {
            for (int w = 0; w < warps_per_token; w++)
                score += s_group_qk[group_idx * warps_per_token + w];
            score *= softmax_scale;
        }
        /* Broadcast score to all lanes in the warp */
        score = __shfl_sync(0xffffffff, score, 0);

        /* --- QJL correction: dot(r_q, unpacked_signs) * norm * sqrt(pi/2) / sqrt(dim) ---
         *
         * r_q (Rademacher-projected query) is precomputed before the token loop.
         * Each warp handles its qblock (32 elements) of the dot product,
         * then we reduce across warps in the group (same as QK reduction). */
        if (qjl_enabled) {
            float norm = __half2float(
                *((const __half *)(qjl_norms_base + token)));

            /* Unpack this lane's sign bit from packed U8 */
            int global_dim_idx = qb * 32 + lane;
            int byte_idx = global_dim_idx / 8;
            int bit_idx  = global_dim_idx % 8;
            const uint8_t *signs = qjl_signs_base + token * signs_per_head;
            float sign_val = ((signs[byte_idx] >> bit_idx) & 1) ? 1.0f : -1.0f;

            /* Partial dot: r_q[qb*32+lane] * sign_val */
            float partial_corr = rq_val * sign_val;

            /* Warp-local reduce (sum 32 lanes) */
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                partial_corr += __shfl_xor_sync(0xffffffff, partial_corr, offset);

            /* Cross-warp reduce within group (sum all qblocks) */
            if (lane == 0)
                s_group_qk[group_idx * warps_per_token + qb] = partial_corr;
            __syncthreads();

            float correction = 0.0f;
            if (lane == 0) {
                for (int w = 0; w < warps_per_token; w++)
                    correction += s_group_qk[group_idx * warps_per_token + w];
                /* Scale: norm * sqrt(π/2) / sqrt(dim) */
                correction *= norm * 1.2533141f * rsqrtf((float)head_dim);
                score += correction * softmax_scale;
            }
            score = __shfl_sync(0xffffffff, score, 0);
            __syncthreads();
        }

        /* --- Online softmax (all warps in group get same score) --- */
        float old_max = group_max;
        group_max = fmaxf(group_max, score);
        float rescale = __expf(old_max - group_max);
        group_sum = group_sum * rescale + __expf(score - group_max);
        out_acc *= rescale;

        /* --- V dequant: each warp dequants its qblock in parallel --- */
        const uint8_t *vp = vi_base + token * packed_dim + qb * bytes_per_qblock;
        float vs = fabsf(__half2float(
            *((const __half *)(vs_base + token * num_qblocks + qb))));

        float v_val = dequant_register(my_packed, vp, vs,
                                       s_codebook, s_sign,
                                       lane, bits, bytes_per_qblock);

        float weight = __expf(score - group_max);
        out_acc += weight * v_val;

        __syncthreads(); /* ensure s_group_qk is safe for next iteration */
    }

    /* --- Write partition results --- */
    /* Each warp writes its qblock portion of the output */
    if (lane == 0) {
        s_group_max[group_idx] = group_max;
        s_group_sum[group_idx] = group_sum;
    }
    /* Each warp in the group writes 32 elements at qb*32 */
    s_group_acc[group_idx * head_dim + qb * 32 + lane] = out_acc;
    __syncthreads();

    /* Group 0 / warp 0 writes the final partition result */
    /* For multiple groups: reduce across groups first */
    if (warp_idx == 0) {
        float block_max = -FLT_MAX;
        for (int g = 0; g < num_token_groups; g++)
            block_max = fmaxf(block_max, s_group_max[g]);

        float block_sum = 0.0f;
        for (int g = 0; g < num_token_groups; g++)
            block_sum += s_group_sum[g] * __expf(s_group_max[g] - block_max);

        /* Reduce accumulators — lane handles one element across all groups */
        /* We need to iterate over all qblocks, but warp 0 has 32 threads = 1 qblock */
        /* So iterate over qblocks */
        int part_idx = head * num_partitions + part;
        if (lane == 0) {
            partial_max[part_idx] = block_max;
            partial_sum[part_idx] = block_sum;
        }

        float *po = partial_out + part_idx * head_dim;
        for (int out_qb = 0; out_qb < num_qblocks; out_qb++) {
            float val = 0.0f;
            for (int g = 0; g < num_token_groups; g++)
                val += s_group_acc[g * head_dim + out_qb * 32 + lane]
                     * __expf(s_group_max[g] - block_max);
            po[out_qb * 32 + lane] = val;
        }
    }
}

/* -----------------------------------------------------------------------
 * Reduce kernel (unchanged)
 * ----------------------------------------------------------------------- */
__global__ void tq_fused_attention_reduce(
    const float *__restrict__ partial_out,
    const float *__restrict__ partial_max,
    const float *__restrict__ partial_sum,
    float       *__restrict__ output,
    int          head_dim,
    int          num_qblocks,
    int          num_partitions)
{
    const int head = blockIdx.x;
    const int lane = threadIdx.x;

    float global_max = -FLT_MAX;
    for (int p = 0; p < num_partitions; p++)
        global_max = fmaxf(global_max, partial_max[head * num_partitions + p]);

    float global_sum = 0.0f;
    float acc[TQ_MAX_QBLOCKS];
    for (int qb = 0; qb < num_qblocks; qb++) acc[qb] = 0.0f;

    for (int p = 0; p < num_partitions; p++) {
        int idx = head * num_partitions + p;
        float rescale = __expf(partial_max[idx] - global_max);
        global_sum += partial_sum[idx] * rescale;
        const float *po = partial_out + idx * head_dim;
        for (int qb = 0; qb < num_qblocks; qb++)
            acc[qb] += po[qb * 32 + lane] * rescale;
    }

    float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
    float *out_head = output + head * head_dim;
    for (int qb = 0; qb < num_qblocks; qb++)
        out_head[qb * 32 + lane] = acc[qb] * inv_sum;
}

/* -----------------------------------------------------------------------
 * Launcher
 * ----------------------------------------------------------------------- */
extern "C" void tq_fused_attention(
    const float    *q,
    const uint8_t  *k_indices,
    const uint16_t *k_scales,
    const uint8_t  *v_indices,
    const uint16_t *v_scales,
    const float    *codebook,
    const float    *sign_pattern,
    float          *output,
    float          *partial_out,
    float          *partial_max_buf,
    float          *partial_sum_buf,
    /* QJL correction (pass nullptr + 0 when disabled) */
    const uint8_t  *qjl_signs,
    const uint16_t *qjl_residual_norms,
    const float    *rq,            /* Rademacher-projected query, or nullptr */
    int32_t         qjl_enabled,
    int32_t         num_attention_heads,
    int32_t         num_kv_heads,
    int32_t         head_dim,
    int32_t         kv_len,
    int32_t         kv_stride,
    int32_t         packed_dim,
    int32_t         num_qblocks,
    int32_t         bits,
    float           softmax_scale,
    int32_t         num_partitions,
    cudaStream_t    stream)
{
    if (kv_len == 0 || num_kv_heads == 0) return;

    dim3 grid1(num_attention_heads, num_partitions);
    int num_token_groups = TQ_NUM_WARPS / num_qblocks;

    size_t smem = sizeof(float) * TQ_MAX_CODEBOOK               /* codebook */
                + sizeof(float) * 32                              /* sign */
                + sizeof(uint8_t) * TQ_NUM_WARPS * 16            /* packed */
                + sizeof(float) * TQ_NUM_WARPS                   /* group_qk */
                + sizeof(float) * num_token_groups                /* group_max */
                + sizeof(float) * num_token_groups                /* group_sum */
                + sizeof(float) * num_token_groups * head_dim;    /* group_acc */

    tq_fused_attention_decode<<<grid1, TQ_NUM_THREADS, smem, stream>>>(
        q, k_indices, k_scales, v_indices, v_scales,
        codebook, sign_pattern,
        partial_out, partial_max_buf, partial_sum_buf,
        qjl_signs, qjl_residual_norms, rq, qjl_enabled,
        num_attention_heads, num_kv_heads,
        head_dim, kv_len, kv_stride, packed_dim, num_qblocks, bits, softmax_scale,
        num_partitions);

    tq_fused_attention_reduce<<<num_attention_heads, 32, 0, stream>>>(
        partial_out, partial_max_buf, partial_sum_buf,
        output,
        head_dim, num_qblocks, num_partitions);
}
