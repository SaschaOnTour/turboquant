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
| Quality Score | 100.0% (rustqual) |
| Tests | 327 |
| Functions | ~343 |
| Dependencies | `half` + `thiserror` (2 total) |

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

turboquant integrates transparently into [mistral.rs](https://github.com/EricLBuehler/mistral.rs) as a KV-cache quantization backend. All models are supported.

```bash
# Run any model with TurboQuant TQ3 KV-cache compression
mistralrs run --pa-cache-type tq3 -m Qwen/Qwen3-0.6B
mistralrs run --pa-cache-type tq4 -m mistralai/Mistral-7B-Instruct-v0.3
```

### Integration Benchmarks (CPU-only, Qwen3-0.6B, 128 decode tokens)

| Context | Variant | Total Time | Prefill tok/s | Decode tok/s | Wall-Clock Overhead |
|---------|---------|-----------|---------------|-------------|---------------------|
| 512 | Normal | 58.4s | 148.1 | 11.8 | — |
| 512 | TQ3 | 64.7s | 141.5 | 9.5 | +11% |
| 2048 | Normal | 2:38 | 58.4 | 11.5 | — |
| 2048 | TQ3 | 2:55 | 59.5 | 7.7 | +10% |
| 4096 | Normal | 7:50 | 32.2 | 10.9 | — |
| 4096 | TQ3 | 8:16 | 31.6 | 6.5 | +6% |
| 16384 | Normal | 1:47:42 | 7.7 | 8.0 | — |
| 16384 | TQ3 | 1:49:00 | 7.6 | 2.9 | +1.2% |

**Key takeaway**: TQ3 overhead **decreases with context length** (11% → 10% → 6% → 1.2%) because prefill dominates at longer contexts and runs at the same speed. The decode throughput difference (dequantization cost) matters less as sequences grow — exactly the regime where KV-cache compression is needed most.

A future GPU kernel implementation (Approach B) would reduce the decode overhead further. See [Approach B Roadmap](../docs/approach-b-roadmap.md).

### Optimizations

The following optimizations were implemented to achieve near-zero overhead:

- **Delta dequantization**: Avoids O(N^2) redundant work by only dequantizing newly added heads
- **Pre-allocated GPU tensor buffer**: Uses `slice_set`/`narrow` for O(1) per-step tensor updates instead of creating new tensors
- **Lazy quantization**: Defers quantization from prefill to first decode step, keeping prefill at full speed
- **Parallel head processing**: Uses rayon for multi-threaded quantization/dequantization across attention heads
- **Batch quantize**: Shares codebook and sign_pattern setup across heads in a batch
- **Zero-copy tensor data extraction**: Extracts tensor data without unnecessary allocations
- **Reusable Vec buffers**: Pre-allocated buffers reused across decode steps to avoid repeated allocation

## Improvements over llama.cpp TurboQuant (tq3_0)

This implementation differs from the [llama.cpp tq3_0 branch](https://github.com/Aaryan-Kapoor/llama.cpp/tree/turboquant-tq3_0) in several important ways:

### 1. QJL Bias Correction (mandatory, not omitted)

llama.cpp tq3_0 implements **only PolarQuant** (Stage 1) and omits QJL entirely. Without QJL, inner product estimates carry a systematic multiplicative bias of `2/pi` that accumulates across all keys in the softmax during attention. This bias is not visible in short-context benchmarks but **degrades quality at long contexts** (8k+ tokens), which is the primary use case for KV-cache compression.

Our implementation includes the full TURBOQUANTprod algorithm (Algorithm 2 from the paper) with QJL bias correction, guaranteeing `E[<y,x>_est] = <y,x>` (mathematically unbiased).

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
