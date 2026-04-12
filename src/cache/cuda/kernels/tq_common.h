/*
 * tq_common.h — Platform-independent TurboQuant algorithm primitives.
 *
 * Contains the core algorithms (unpack, WHT, codebook, sign pattern)
 * shared between CUDA and future Metal kernels. No CUDA or Metal API
 * calls — only plain C arithmetic.
 *
 * Golden ratio constant for sign pattern generation.
 */

#ifndef TQ_COMMON_H
#define TQ_COMMON_H

#include <stdint.h>

/* -----------------------------------------------------------------------
 * Constants
 * ----------------------------------------------------------------------- */

#define TQ_GOLDEN_RATIO 0x9E3779B97F4A7C15ULL

/* Maximum supported head dimension (power of two). */
#define TQ_MAX_HEAD_DIM 1024

/* Maximum packed bytes per block (4-bit, dim=1024 → 512 bytes). */
#define TQ_MAX_PACKED_BYTES 512

/* -----------------------------------------------------------------------
 * 2-bit unpacking: 4 indices per byte
 *
 * Byte layout: [idx0:2 | idx1:2 | idx2:2 | idx3:2]
 *              bits 1-0  bits 3-2  bits 5-4  bits 7-6
 * ----------------------------------------------------------------------- */

static inline __device__ __host__
uint8_t tq_unpack_2bit(const uint8_t *packed, int tid) {
    return (packed[tid >> 2] >> ((tid & 3) << 1)) & 0x3;
}

/* -----------------------------------------------------------------------
 * 3-bit unpacking: 8 indices per 3 bytes
 *
 * Matches the exact layout in turboquant/src/packed.rs:348-358.
 * ----------------------------------------------------------------------- */

static inline __device__ __host__
uint8_t tq_unpack_3bit(const uint8_t *packed, int tid) {
    int group = tid >> 3;       /* tid / 8 */
    int pos   = tid & 7;        /* tid % 8 */
    int base  = group * 3;
    uint8_t b0 = packed[base];
    uint8_t b1 = packed[base + 1];
    uint8_t b2 = packed[base + 2];

    switch (pos) {
        case 0: return  b0        & 0x7;
        case 1: return (b0 >> 3)  & 0x7;
        case 2: return ((b0 >> 6) & 0x3) | ((b1 & 0x1) << 2);
        case 3: return (b1 >> 1)  & 0x7;
        case 4: return (b1 >> 4)  & 0x7;
        case 5: return ((b1 >> 7) & 0x1) | ((b2 & 0x3) << 1);
        case 6: return (b2 >> 2)  & 0x7;
        case 7: return (b2 >> 5)  & 0x7;
        default: return 0; /* unreachable */
    }
}

/* -----------------------------------------------------------------------
 * 4-bit unpacking: 2 indices per byte
 * ----------------------------------------------------------------------- */

static inline __device__ __host__
uint8_t tq_unpack_4bit(const uint8_t *packed, int tid) {
    return (packed[tid >> 1] >> ((tid & 1) << 2)) & 0xF;
}

/* -----------------------------------------------------------------------
 * Golden ratio hash for sign pattern generation.
 *
 * sign[i] = +1.0 if ((seed + i) * GOLDEN_RATIO) has even LSB, else -1.0
 * ----------------------------------------------------------------------- */

static inline __device__ __host__
float tq_sign(uint64_t seed, int index) {
    uint64_t combined = (seed + (uint64_t)index) * TQ_GOLDEN_RATIO;
    return (combined & 1ULL) == 0 ? 1.0f : -1.0f;
}

/* -----------------------------------------------------------------------
 * Atomic byte-OR helper.
 *
 * On CUDA device: uses atomicOr on the containing uint32 word.
 * On host: plain OR (single-threaded, no atomics needed).
 * ----------------------------------------------------------------------- */

static inline __device__ __host__
void tq_atomic_byte_or(uint8_t *packed, int byte_idx, uint32_t bits) {
#ifdef __CUDA_ARCH__
    if (bits) {
        atomicOr((unsigned int *)(packed + (byte_idx & ~3)),
                 bits << ((byte_idx & 3) * 8));
    }
#else
    packed[byte_idx] |= (uint8_t)bits;
#endif
}

/* -----------------------------------------------------------------------
 * 2-bit packing: 4 indices per byte (for quantize kernel)
 *
 * Thread `tid` contributes its index to the correct byte position.
 * All threads in a group of 4 must cooperate to build one byte.
 * ----------------------------------------------------------------------- */

static inline __device__ __host__
void tq_pack_2bit(uint8_t *packed, int tid, uint8_t idx) {
    int byte_idx = tid >> 2;
    int shift = (tid & 3) << 1;
    tq_atomic_byte_or(packed, byte_idx, (uint32_t)((idx & 0x3) << shift));
}

/* -----------------------------------------------------------------------
 * 3-bit packing: 8 indices per 3 bytes (for quantize kernel)
 *
 * Matches the exact bit layout of tq_unpack_3bit above.
 * Thread `tid` contributes its 3-bit index to the correct bit positions
 * across up to 2 bytes. Uses tq_atomic_byte_or for thread-safe writes.
 * ----------------------------------------------------------------------- */

static inline __device__ __host__
void tq_pack_3bit(uint8_t *packed, int tid, uint8_t idx) {
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
    tq_atomic_byte_or(packed, base,     b0_bits);
    tq_atomic_byte_or(packed, base + 1, b1_bits);
    tq_atomic_byte_or(packed, base + 2, b2_bits);
}

/* -----------------------------------------------------------------------
 * 4-bit packing: 2 indices per byte (for quantize kernel)
 * ----------------------------------------------------------------------- */

static inline __device__ __host__
void tq_pack_4bit(uint8_t *packed, int tid, uint8_t idx) {
    int byte_idx = tid >> 1;
    int shift = (tid & 1) << 2;
    tq_atomic_byte_or(packed, byte_idx, (uint32_t)((idx & 0xF) << shift));
}

/* -----------------------------------------------------------------------
 * SplitMix64 hash for QJL Rademacher sign generation.
 *
 * Returns +1.0 or -1.0 based on the LSB of the hash output.
 * Matches turboquant/src/qjl.rs rademacher_sign_from_hash().
 * ----------------------------------------------------------------------- */

#define TQ_SPLITMIX_GAMMA      0x9E3779B97F4A7C15ULL
#define TQ_SPLITMIX_MUL_1      0xBF58476D1CE4E5B9ULL
#define TQ_SPLITMIX_MUL_2      0x94D049BB133111EBULL
#define TQ_SEED_MIX_MULTIPLIER 0x517CC1B727220A95ULL
#define TQ_SEED_MIX_XOR        0x6C62272E07BB0142ULL
#define TQ_SEED_MIX_SHIFT      32

/* Per-row seed mixing — matches turboquant/src/qjl.rs mix_seed(). */
static inline __device__ __host__
uint64_t tq_mix_seed(uint64_t seed, int row) {
    uint64_t mixed = seed * TQ_SEED_MIX_MULTIPLIER + (uint64_t)row;
    return (mixed ^ (mixed >> TQ_SEED_MIX_SHIFT)) ^ TQ_SEED_MIX_XOR;
}

static inline __device__ __host__
float tq_rademacher_sign(uint64_t seed, int row, int col) {
    uint64_t row_seed = tq_mix_seed(seed, row);
    uint64_t z = row_seed
        + ((uint64_t)row * TQ_SPLITMIX_GAMMA)
        + (uint64_t)col;
    z = (z ^ (z >> 30)) * TQ_SPLITMIX_MUL_1;
    z = (z ^ (z >> 27)) * TQ_SPLITMIX_MUL_2;
    z = z ^ (z >> 31);
    return (z & 1ULL) == 0 ? 1.0f : -1.0f;
}

#endif /* TQ_COMMON_H */
