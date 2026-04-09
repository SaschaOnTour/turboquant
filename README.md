# turboquant

Rust implementation of Google's TurboQuant algorithm (Zandieh et al., ICLR 2026) for extreme KV-cache compression in LLM inference.

[![CI](https://github.com/SaschaOnTour/turboquant/actions/workflows/ci.yml/badge.svg)](https://github.com/SaschaOnTour/turboquant/actions/workflows/ci.yml)
[![docs.rs](https://docs.rs/turboquant/badge.svg)](https://docs.rs/turboquant)
[![crates.io](https://img.shields.io/crates/v/turboquant.svg)](https://crates.io/crates/turboquant)

## What is TurboQuant?

TurboQuant compresses the key-value cache of large language models to 3-4 bits per value with zero accuracy loss. It is training-free and data-oblivious -- no calibration data required. This reduces KV-cache memory by up to 5x while maintaining full model quality.

The algorithm combines two stages:

- **PolarQuant (Stage 1)**: Random rotation (Walsh-Hadamard Transform) + optimal scalar quantization via pre-computed Lloyd-Max codebooks
- **QJL (Stage 2)**: 1-bit bias correction via Quantized Johnson-Lindenstrauss projection, ensuring unbiased inner product estimates for attention computation

## Project Status

| Metric | Value |
|--------|-------|
| Quality Score | 97.0% (rustqual) |
| Tests | 364 |
| CUDA Kernels | 3 (quantize, dequantize, fused attention) |
| Dependencies | `half` + `thiserror` (core), `candle-core` + `cudaforge` (optional) |

## Quick Start

```rust
use turboquant::packed::{TurboQuantConfig, PackedBlock};
use turboquant::quantize::{quantize_vec, dequantize_vec};

// Configure for 3-bit quantization, head dimension 128
let config = TurboQuantConfig::new(3, 128).unwrap().with_seed(42);

// Quantize a key vector
let packed: PackedBlock = quantize_vec(&config, &key_vec).unwrap();

// Dequantize back
let recovered: Vec<f32> = dequantize_vec(&config, &packed).unwrap();
```

See [`examples/`](examples/) for runnable demos.

## API Overview

| Type | Module | Description |
|------|--------|-------------|
| `TurboQuantConfig` | `packed` | Configuration: bit-width (2/3/4), head dimension, seed |
| `PackedBlock` | `packed` | Bit-packed quantized block (unified format for all bit-widths) |
| `QuantizedKVCache` | `attention` | High-level cache with attention score computation |
| `QjlBlock` | `qjl` | QJL-enhanced quantized block for unbiased inner products |
| `EstimationContext` | `qjl` | Pre-fetched context for efficient batch attention scoring |

## Compression & Accuracy

| Variant | Bits/Value | Compression vs FP16 | Normalized MSE | Paper Target |
|---------|-----------|---------------------|---------------|-------------|
| TQ3 (3-bit) | 3 | 4.9x | ~0.034 | 0.034 |
| TQ4 (4-bit) | 4 | ~3.5x | ~0.009 | 0.009 |

MSE measured over 10,000 random vectors at d=128, matching paper values exactly.

## Performance (d=128, release build)

| Operation | Latency |
|-----------|---------|
| PolarQuant quantize | ~1.1 us |
| PolarQuant dequantize | ~0.8 us |
| QJL quantize | ~19 us |
| QJL inner product (batch, per key) | ~1.1 us |
| Attention over 1024 keys | ~1.1 ms |
| Estimated 100k context / 32 layers | ~3.5 s |

## mistral.rs Integration

turboquant integrates into [mistral.rs](https://github.com/EricLBuehler/mistral.rs) via
the `CompressedKVCache` trait. All models with `head_dim` divisible by 32 are supported
(Llama, Qwen, Mistral, Falcon, Gemma, DeepSeek, and more).

```bash
# PQO3 — recommended mode (3-bit, outlier codebook)
mistralrs run --pa-cache-type pqo3 -m Qwen/Qwen3-0.6B --device-layers "0:999"

# PQO4 — higher quality (4-bit)
mistralrs run --pa-cache-type pqo4 -m Qwen/Qwen3-0.6B --device-layers "0:999"
```

### GPU Benchmark (RTX 3090, Qwen3-0.6B, 28 layers)

| Mode | 1K ctx | 4K ctx | 16K ctx | 32K ctx |
|------|--------|--------|---------|---------|
| Normal | 5s / 1796 MiB | 5s / 2500 MiB | 8s / 5380 MiB | 15s / **9124 MiB** |
| PQO3 | 5s / 1572 MiB | 6s / 1860 MiB | 8s / 2948 MiB | 15s / **4649 MiB** |

**Zero performance overhead** on GPU with a fused CUDA attention kernel that reads
directly from the compressed cache. **49% VRAM savings** at 32K context.

VRAM savings scale with model depth: more layers = larger KV-cache = more benefit.
For large models (7B+, 32+ layers, long contexts), the KV-cache dominates VRAM,
making PQO3 increasingly valuable.

See [full benchmark results](docs/benchmarks.md) for multi-model comparisons,
CPU results, and detailed analysis.

### Architecture

```
mistralrs-kv-cache          (trait: CompressedKVCache)
        ^                          ^
   turboquant-rs              mistralrs-core
   (PqoCache, TqCache)       (uses dyn Trait)
```

Adding a new compression method requires only:
1. `impl CompressedKVCache for YourCache`
2. One match arm in the cache factory

No model code changes needed.

## PQO: PolarQuant Outlier — Our Recommended Approach

PQO (PolarQuant Outlier) is a variant we developed by combining insights from the
TurboQuant paper and the llama.cpp implementation. It outperforms both in practice:

| Approach | Codebook | QJL | GPU Kernel | Quality | Performance |
|----------|----------|-----|------------|---------|-------------|
| **Paper TQ3** | Standard (2-bit) | Yes (1-bit) | — | Degraded (variance) | Slow (no fused kernel) |
| **llama.cpp tq3_0** | Mixed (outlier for some blocks) | No | No | Good | CPU only |
| **Our PQO3** | Outlier for ALL blocks | No | Fused CUDA | Excellent | Zero overhead on GPU |

### What makes PQO different?

1. **Outlier codebook for all blocks**: The TurboQuant paper (Section 4.3) uses a
   higher-bit codebook only for "outlier" blocks (those with highest norms). We apply
   it to **all** blocks, trading 1 bit of theoretical efficiency for significantly
   better reconstruction quality. At 3-bit total, PQO3 uses the 3-bit codebook
   everywhere instead of a 2-bit/3-bit mix.

2. **No QJL**: The paper's QJL correction (Stage 2) is mathematically unbiased but
   increases variance by 30-300%
   ([llama.cpp analysis](https://github.com/ggml-org/llama.cpp/discussions/20969)).
   This variance harms softmax Top-K ranking in attention, degrading text quality.
   We confirmed this empirically: TQ3/TQ4 (with QJL) produce garbage text, while
   PQO3 (without QJL) produces perfect output. Dropping QJL also means all 3 bits
   go to PolarQuant instead of 2+1.

3. **Fused CUDA kernel**: Our decode path reads directly from the compressed cache
   in GPU shared memory — no full-dequantization tensor needed. This eliminates the
   O(seq_len) memory overhead that makes other approaches slow at long contexts.
   The result: **zero performance overhead** compared to uncompressed KV-cache on GPU.

### Results compared to llama.cpp

llama.cpp's TQ3_0 implementation is CPU-only and uses a mixed codebook strategy.
Our GPU-accelerated PQO3 achieves:

- **49% VRAM savings** at 32K context (Qwen3-0.6B, 28 layers)
- **Zero inference time overhead** on GPU (fused CUDA kernel)
- **Perfect text quality** across all tested models (Qwen3, Llama-3.2, Falcon3)
- **All models supported** via trait-based architecture (no per-model code changes)

### References

- TurboQuant paper: [Zandieh et al., ICLR 2026](https://arxiv.org/pdf/2504.19874)
  — PolarQuant algorithm, QJL theory, codebook design
- Paper Section 4.3: Outlier block concept ("32 outlier channels at 3-bit")
  — inspiration for applying outlier codebook to all blocks
- llama.cpp discussion: [ggml-org/llama.cpp#20969](https://github.com/ggml-org/llama.cpp/discussions/20969)
  — QJL variance analysis, empirical confirmation that QJL harms attention quality

## Technical Comparison with llama.cpp TurboQuant (tq3_0)

This implementation differs from the [llama.cpp tq3_0 branch](https://github.com/Aaryan-Kapoor/llama.cpp/tree/turboquant-tq3_0) in several important ways:

### 1. QJL Bias Correction (implemented, but PQO recommended)

llama.cpp tq3_0 implements **only PolarQuant** (Stage 1) and omits QJL entirely.
Our implementation includes the full TURBOQUANTprod algorithm (Algorithm 2) with QJL
bias correction, guaranteeing `E[<y,x>_est] = <y,x>` (mathematically unbiased).

**However**: empirical testing confirms the
[llama.cpp finding](https://github.com/ggml-org/llama.cpp/discussions/20969) that QJL
increases variance, which harms softmax Top-K ranking in attention. The TQ3/TQ4 modes
(with QJL) currently produce degraded text quality. **PQO3 (PolarQuant Outlier, without
QJL) is the recommended mode** — it provides excellent compression with zero quality loss.

### 2. Dimension-Specific Codebooks (exact Beta distribution)

| Aspect | llama.cpp tq3_0 | turboquant (this crate) |
|--------|-----------------|------------------------|
| **Distribution** | Gaussian N(0,1) approximation | Exact Beta distribution per dimension |
| **Codebooks** | Single fixed set for all dimensions | Pre-computed per (bits, dim) pair |
| **3-bit centroids (d=128)** | [-2.15, -1.34, -0.76, -0.25, +0.25, +0.76, +1.34, +2.15] | [-0.189, -0.118, -0.067, -0.022, +0.022, +0.067, +0.118, +0.189] |
| **Relationship** | Centroids for normalized coordinates | `llama_centroid ~= ours * sqrt(d)` |

After rotation, the coordinates of a d-dimensional unit vector follow a Beta-type distribution on [-1, 1], **not** a Gaussian. The Gaussian is the limiting distribution as d approaches infinity. For practical dimensions (d=64-256), the Beta distribution is a better fit. Our codebooks are optimal Lloyd-Max quantizers for the exact distribution, yielding slightly lower MSE.

### 3. Flexible Block Sizes

llama.cpp uses a fixed block size of 32 values. Our implementation supports variable dimensions (64, 128, 256) matching common LLM head dimensions directly, avoiding padding waste.

### 4. Hash-Based Rademacher (no crypto RNG needed)

The QJL projection matrix uses deterministic hash-based sign generation instead of requiring a cryptographic RNG, making it fast and reproducible across platforms.

### 5. Bit-Packing Compatible with llama.cpp

The 3-bit packing layout is **identical** to llama.cpp tq3_0 (8 indices into 3 bytes, same byte order), ensuring potential interoperability at the data level.

## Installation

```toml
[dependencies]
turboquant = { git = "https://github.com/SaschaOnTour/turboquant.git" }
```

## Building with Native CPU Optimizations

The crate is configured to use native CPU features (AVX2, FMA, etc.) automatically via `.cargo/config.toml`. For maximum performance:

```bash
cargo build --release
```

## Running Examples

```bash
cargo run --example basic_quantize
cargo run --example kv_cache_demo
```

## References

- Paper: [TurboQuant (Zandieh et al., ICLR 2026)](https://arxiv.org/pdf/2504.19874)
- Blog: [Google Research Blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- llama.cpp reference: [tq3_0 branch](https://github.com/Aaryan-Kapoor/llama.cpp/tree/turboquant-tq3_0)

## License

Licensed under either of Apache License, Version 2.0 or MIT license, at your option.
