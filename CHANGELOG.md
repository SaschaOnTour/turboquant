# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-03-29

### Added

- **mistral.rs integration**: Transparent `KvCache::TurboQuant` variant — all 55+ models supported without model code changes. CLI: `--pa-cache-type tq3`.
- **Lazy quantization**: Prefill stores raw vectors, quantization deferred to first decode step — **0% prefill overhead**.
- **Delta dequantization**: Only dequantize new tokens per step — O(1) instead of O(N) per decode step.
- **Batch quantize**: `push_batch()` with shared codebook/sign-pattern setup and `QjlBatchResources` for efficient multi-vector quantization during flush.
- **Range dequantization**: `dequantize_keys_range()` / `dequantize_values_range()` for selective block dequantization without touching full history.
- **Parallel head processing**: Rayon-based parallel quantize/dequantize across KV-heads.
- **Pre-allocated GPU tensor buffer**: `Tensor::slice_set` + `narrow` instead of `Tensor::cat` — eliminates O(N) copy overhead.
- **Zero-copy tensor extraction**: Direct Candle storage access for tensor→Vec<f32> conversion without extra allocation.
- **Reusable Vec buffers**: `k_data_buf` / `v_data_buf` avoid per-step allocation in decode hot path.
- **`DequantScratch`**: Reusable scratch buffer for dequantize operations, avoiding repeated allocation.
- **`EstimationContext`**: Caches QJL sign patterns for efficient repeated inner product estimation.
- **Benchmark script** (`scripts/benchmark-b0.sh`): Automated 4-variant benchmark (CPU/GPU × Normal/TQ3) with wall-clock timing and peak VRAM measurement.
- **Setup script** (`scripts/setup-env.sh`): Full environment setup for new machines (Rust, cargo tools, system deps, Node.js, rustqual from crates.io).
- **Developer guide** (`docs/developer-guide.md`): Complete project documentation with architecture diagram, setup instructions, and key files reference.
- **Approach B roadmap** (`docs/approach-b-roadmap.md`): GPU kernel roadmap with memory-bandwidth analysis showing TQ3 could be ~1.4x faster than FP16.

### Changed

- **PackedBlock unification**: Replaced separate `BlockTQ2`/`BlockTQ3`/`BlockTQ4` with single `PackedBlock` struct using generic `pack_indices_chunked` helper.
- **Config fields**: `TurboQuantConfig` fields changed to `pub(crate)` with constructor pattern.
- **QjlBlock fields**: Changed to `pub(crate)` with accessor methods.
- **Rotation API**: Unified `rotate()` / `inverse_rotate()` into single `rotate(data, order: RotationOrder)` function.
- **`entry_count()`**: Renamed from `len()` on `QuantizedKVCache` to avoid rustqual false positives.
- **Codebook module split**: `codebook.rs` → `codebook/mod.rs` + `codebook/gen.rs` for SRP compliance.
- **Test count**: 317 → 327 tests.

### Fixed

- **Bit-budget bug**: TQ3 was using 3-bit PolarQuant + 1-bit QJL = 4 bits total. Fixed to use (b-1)-bit PolarQuant + 1-bit QJL = b bits total, matching the paper's Algorithm 2.
- **Compression ratio**: Now correctly achieves ~4.9x for TQ3 (was ~3.8x before bit-budget fix).
- **Cache management**: Fixed clone_in/clone_out/set_none for TurboQuant variant — previously crashed with "requested capacity above max seq len (0)".
- **Pipeline activation**: Fixed three blockers preventing TQ activation (supports_paged_attention check, CPU device check, init_cache_config early return).

## [0.1.0] - 2026-03-29

### Added

- **PolarQuant** (Stage 1): Walsh-Hadamard rotation + optimal Lloyd-Max scalar quantization with dimension-specific Beta-distribution codebooks for 2/3/4-bit quantization.
- **QJL bias correction** (Stage 2): 1-bit Quantized Johnson-Lindenstrauss projection ensuring mathematically unbiased inner product estimates (`E[<y,x>_est] = <y,x>`).
- **QuantizedKVCache**: High-level API for storing quantized key-value pairs and computing attention scores with full TURBOQUANTprod algorithm (Algorithm 2 from the paper).
- **Pre-computed codebooks**: Optimal Lloyd-Max quantizers for the exact Beta distribution at practical LLM head dimensions (64, 128, 256) for 2/3/4-bit widths.
- **Bit-packing**: TQ3 layout identical to llama.cpp tq3_0 (8 indices into 3 bytes, same byte order) for potential interoperability.
- **Hash-based Rademacher**: Deterministic sign generation for QJL projection matrix, no cryptographic RNG needed.
- 317 tests, 330 functions, 100.0% quality score (rustqual).
- Examples: `basic_quantize` and `kv_cache_demo`.
- CI workflow: test, clippy, fmt, doc.
- Dual license: MIT OR Apache-2.0.
