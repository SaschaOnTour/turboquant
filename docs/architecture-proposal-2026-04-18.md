# Architektur-Vorschlag: turboquant-rs — Clean Architecture + DDD Refactoring

**Autor:** Senior Architect Review · **Stand:** 2026-04-18
**Scope:** `/workspace/turboquant` (v0.3.1, ~8900 LOC, 29 Dateien)

## TL;DR

Der Kern der Crate (Quantisierungsmathematik) ist bereits sauber getrennt. Der Schmerzpunkt ist die **Integrations-Schicht `cache/`**, die vier Concerns vermischt: Cache-Lifecycle, Candle-Tensor-Integration, GPU-Dispatch via `#[cfg]`-Flags, und rohe CUDA-FFI-Aufrufe. Vorschlag: **Hexagonale Architektur (Ports & Adapters)** mit strikter Dependency-Regel und einer einzigen neuen Abstraktion — `ComputeBackend`-Trait — die die gesamte CPU/CUDA-Dispatch-Logik kapselt.

Der Vorschlag ist **inkrementell migrierbar** in 5 Phasen, jede einzeln mergebar.

---

## 1. Befund: Was ist gut, was ist kaputt

### ✅ Das funktioniert bereits
- **Purer Kern**: `quantize.rs`, `qjl.rs`, `codebook/`, `rotation.rs`, `packing/`, `math.rs` — kein Candle, kein CUDA, keine Framework-Abhängigkeit
- **Error-Typ** zentral und konsistent (`error::TurboQuantError`)
- **Trait-basierte Consumer-API** (`CompressedKVCache` in `mistralrs-kv-cache`) ist schon ein sauberer Port
- **Config-DTOs** (`TurboQuantConfig`, `CacheConfig`) sind stabile Value-Objects

### ❌ Das ist kaputt

| # | Befund | Datei | Schweregrad |
|---|---|---|---|
| 1 | **God-Struct `PqoCache`** — 18 Methoden, mischt Lifecycle + Storage + Quantisierung + GPU-Dispatch + Attention | `cache/pqo.rs` | Hoch |
| 2 | **Upward-Dependency**: Domain-Code ruft Adapter-Code auf | `cache/quantize_tensor.rs:59` ruft `cuda::quantize::cuda_quantize_fast()` | Hoch |
| 3 | **`#[cfg(feature = "cuda")]` in Domain-Code** statt am Adapter-Rand | `cache/quantize_tensor.rs:54`, `cache/pqo.rs:214` | Hoch |
| 4 | **Duplizierte 84-Zeilen-Pointer-Extraktion** | `cache/cuda/attention.rs:81-139`, `cache/cuda/quantize.rs` | Mittel |
| 5 | **1646-Zeilen-Datei `attention.rs`** vermischt CPU-Cache + Attention-Score-Berechnung + Dequantisierungs-Pfade | `attention.rs` | Mittel |
| 6 | **1487-Zeilen-Datei `qjl.rs`** ist eigentlich zwei Domänen (Projektion + Estimator) | `qjl.rs` | Mittel |
| 7 | **Monolithischer Factory-Call `GpuPrecomputed::new()`** — keine Erweiterungspunkte | `cache/precomputed/mod.rs:45-98` | Niedrig |

---

## 2. Zielbild — Hexagonale Architektur (Ports & Adapters)

```
┌────────────────────────────────────────────────────────────────┐
│  BRIDGES  (Consumer-Glue; impl mistralrs_kv_cache::*)          │  Layer 4
│  - PqoCache, TqCache, CacheConfig                              │
└────────────────────────────────────────────────────────────────┘
         │ depends on
         ▼
┌────────────────────────────────────────────────────────────────┐
│  ADAPTERS  (konkrete Implementierungen der Ports)              │  Layer 3
│  - candle/ (Tensor-Marshalling)                                │
│  - backend_cpu/ (ComputeBackend für CPU)                       │
│  - backend_cuda/ (ComputeBackend + FFI, einziger Ort mit cuda) │
│  - precomputed/ (GPU-Precompute-Builder)                       │
└────────────────────────────────────────────────────────────────┘
         │ depends on
         ▼
┌────────────────────────────────────────────────────────────────┐
│  PORTS  (Trait-Abstraktionen nach außen)                       │  Layer 2
│  - ComputeBackend (quantize, dequantize, fused_attention)      │
│  - TensorView (minimal tensor port; optional)                  │
└────────────────────────────────────────────────────────────────┘
         │ depends on
         ▼
┌────────────────────────────────────────────────────────────────┐
│  APPLICATION  (Use-Cases, device-agnostisch)                   │  Layer 1
│  - quantize_block, dequantize_block                            │
│  - estimate_attention (QJL-bewusst)                            │
│  - cache_lifecycle (Prefill/Decode/Reset State)                │
└────────────────────────────────────────────────────────────────┘
         │ depends on
         ▼
┌────────────────────────────────────────────────────────────────┐
│  DOMAIN  (Value Objects, Entities, reine Domänen-Services)     │  Layer 0
│  - BitWidth, BlockShape, QuantNormMode (VO)                    │
│  - Codebook, QuantizedBlock, PackedIndices (Entities)          │
│  - Rotation, QjlProjection (Domain Services, reines Math)      │
└────────────────────────────────────────────────────────────────┘
```

**Dependency-Regel (strikt):** Pfeile zeigen nach unten. Ein unteres Layer darf **nie** über ein oberes Layer Bescheid wissen. `#[cfg(feature = "cuda")]` existiert **nur** im `adapters/backend_cuda/`-Zweig.

---

## 3. Zielverzeichnisstruktur

```
src/
├── lib.rs                          (Re-exports; Feature-Gates)
│
├── domain/                         Layer 0 — pure, zero framework dep
│   ├── mod.rs
│   ├── value_objects.rs            BitWidth, BlockShape, QuantNormMode
│   ├── errors.rs                   (aktuell error.rs)
│   ├── codebook/
│   │   ├── mod.rs                  Codebook entity, StaticCodebook
│   │   ├── generator.rs            Lloyd-Max (aktuell codebook/gen.rs)
│   │   ├── tables.rs               (aktuell codebook/tables.rs)
│   │   └── lookup.rs               Binary search
│   ├── rotation/
│   │   ├── mod.rs
│   │   ├── hadamard.rs             WHT in-place (aus rotation.rs)
│   │   └── signs.rs                Rademacher sign pattern
│   ├── packing/
│   │   ├── mod.rs                  PackedBlock entity (aus packed/mod.rs)
│   │   ├── pack.rs                 2/3/4-bit pack
│   │   ├── unpack.rs
│   │   └── indices.rs              (aktuell packed/indices.rs)
│   ├── qjl/
│   │   ├── mod.rs
│   │   ├── projection.rs           1-bit JL signs
│   │   └── estimator.rs            Inner-product estimation
│   └── math.rs                     (aktuell math.rs; Lanczos, Simpson)
│
├── application/                    Layer 1 — use cases, device-agnostisch
│   ├── mod.rs
│   ├── quantize_block.rs           rotate → lookup → pack (aus quantize.rs)
│   ├── dequantize_block.rs         unpack → codebook → inverse rotation
│   ├── attention_estimate.rs       (aus qjl.rs::estimate_*; aus attention.rs)
│   └── cache_state.rs              Prefill/Decode/Reset-Logik, ohne Tensor
│
├── ports/                          Layer 2 — Trait-Abstraktionen
│   ├── mod.rs
│   ├── compute_backend.rs          trait ComputeBackend
│   ├── tensor_view.rs              minimal tensor port (optional)
│   └── cache.rs                    re-export von CompressedKVCache
│
├── adapters/                       Layer 3 — konkrete Impls
│   ├── mod.rs
│   ├── candle/                     Candle-Tensor ↔ domain
│   │   ├── mod.rs
│   │   ├── quantize_tensor.rs      (aus cache/quantize_tensor.rs, OHNE cuda)
│   │   ├── wht_tensor.rs           (aus cache/wht_tensor.rs)
│   │   └── storage.rs              (aus cache/storage.rs)
│   ├── backend_cpu/                ComputeBackend impl (CPU)
│   │   └── mod.rs
│   ├── backend_cuda/               ComputeBackend impl (CUDA) — EINZIGER Ort mit cuda-Feature
│   │   ├── mod.rs
│   │   ├── ffi.rs                  (aus cache/cuda/ffi.rs)
│   │   ├── tensor_ptr.rs           NEU: dedupliziert den 84-Zeilen-Unpack
│   │   ├── quantize.rs             (aus cache/cuda/quantize.rs, schlank)
│   │   ├── dequantize.rs
│   │   └── fused_attention.rs      (aus cache/cuda/attention.rs, schlank)
│   └── precomputed/                GPU-Precompute als Builder-Pattern
│       ├── mod.rs                  PrecomputeBuilder (Chainable)
│       ├── rotation_table.rs
│       ├── codebook_table.rs
│       └── qjl_signs.rs
│
└── bridges/                        Layer 4 — Consumer-Glue
    ├── mod.rs
    ├── pqo_cache.rs                impl CompressedKVCache (thin delegate)
    ├── tq_cache.rs                 impl CompressedKVCache (thin delegate)
    └── config.rs                   (aus cache/config.rs)
```

---

## 4. Kern-Abstraktion: `ComputeBackend` Trait

Die einzige neue architektonische Entscheidung. Alles andere folgt.

```rust
// src/ports/compute_backend.rs

/// Abstracts the hardware path for quantize / dequantize / fused attention.
/// CPU and CUDA impls live in adapters/backend_{cpu,cuda}/.
pub trait ComputeBackend: Send + Sync {
    /// Quantize a K or V block tensor into packed indices + scales.
    fn quantize(
        &self,
        input: &Tensor,
        n_blocks: usize,
        config: &QuantConfig,
    ) -> Result<QuantizedBlocks>;

    /// Reconstruct a K or V block tensor from packed indices + scales.
    fn dequantize(
        &self,
        qt: &QuantizedBlocks,
        target_dtype: DType,
    ) -> Result<Tensor>;

    /// Try a fused attention kernel. Return None when the backend
    /// has no such kernel (caller must fall back to dequant + SDPA).
    fn fused_attention(
        &self,
        qt: &QuantizedBlocks,
        q: &Tensor,
        config: &AttendConfig,
    ) -> Result<Option<Tensor>>;
}
```

**Effekte:**
- `PqoCache::decode()` wird ~15 Zeilen: `backend.fused_attention(...)?` oder Fallback auf `backend.dequantize(...)`
- `#[cfg(feature = "cuda")]` verschwindet aus `cache/pqo.rs` komplett
- Testbarkeit: `MockComputeBackend` für Unit-Tests ohne GPU
- Erweiterbarkeit: Metal/Vulkan-Backend = neue Adapter-Crate, kein Core-Change

---

## 5. Weitere Schlüssel-Designentscheidungen

### 5.1 `PqoCache`/`TqCache` werden thin bridges

```rust
// src/bridges/pqo_cache.rs
pub struct PqoCache {
    state: CacheState,              // Layer 1 — reine Zustandslogik
    storage: CandleStorage,         // Layer 3 — Tensor-Buffer pro Layer
    backend: Arc<dyn ComputeBackend>, // Layer 2 — Dispatch
}

impl CompressedKVCache for PqoCache {
    fn decode(&mut self, layer: usize, k: &Tensor, v: &Tensor, q: &Tensor, cfg: &AttendConfig)
        -> Result<DecodeOutput>
    {
        let qt = self.backend.quantize(&concat(k, v), ...)?;
        self.storage.append(layer, qt)?;

        if let Some(out) = self.backend.fused_attention(&self.storage.get(layer)?, q, cfg)? {
            return Ok(DecodeOutput::Fused(out));
        }
        let (k_full, v_full) = self.backend.dequantize(...)?;
        Ok(DecodeOutput::Dequantized(DequantResult { k: k_full, v: v_full, logit_bias: None }))
    }
    // ... prefill, reset, seq_len delegate to state + storage + backend
}
```

**Ergebnis:** `cache/pqo.rs` schrumpft von **236 Zeilen mit 7 Verantwortlichkeiten** auf ~**80 Zeilen mit 1 Verantwortung** (CompressedKVCache-Trait-Implementierung).

### 5.2 `attention.rs` wird zerlegt

Aktuell 1646 Zeilen. Zerlegung nach Responsibility:

| Zielort | Inhalt | Lines (geschätzt) |
|---|---|---|
| `domain/qjl/estimator.rs` | `estimate_inner_product_with_codebook` | ~200 |
| `application/attention_estimate.rs` | Aufruf-Orchestrierung | ~100 |
| `adapters/candle/cpu_kv_cache.rs` | `QuantizedKVCache` (CPU-Struct) | ~800 |
| `application/cache_state.rs` | Dequant-Scratch-Management | ~300 |
| ENTFÄLLT | Duplikate, Kommentare | ~250 |

### 5.3 `GpuPrecomputed::new()` wird Builder

Statt monolithisch:
```rust
let precomputed = PrecomputeBuilder::new(&config, &device)
    .with_rotation()?
    .with_codebooks(QuantNormMode::MaxNorm)?
    .with_qjl_signs(seed)?
    .build()?;
```

Jeder Step ist einzeln testbar, lazy-loadable, austauschbar.

### 5.4 Dedup des Pointer-Unpack

Neu: `adapters/backend_cuda/tensor_ptr.rs`:
```rust
pub(crate) struct CudaPtr<T> { ... }
impl<T> CudaPtr<T> {
    pub fn from_tensor(t: &Tensor) -> Result<Self> { /* 15 Zeilen */ }
    pub fn as_raw(&self) -> *const T { ... }
}
```

`fused_attention.rs` und `quantize.rs` reduzieren sich von je ~170 Zeilen auf ~40 Zeilen.

### 5.5 Feature-Gates konsolidieren

Nach Refactoring existieren `#[cfg(feature = "cuda")]` nur noch in:
- `src/adapters/backend_cuda/**` (gesamtes Modul)
- `src/lib.rs` (conditional `pub use adapters::backend_cuda`)

Vorher: **9+ verstreute cfg-Gates über die ganze Crate**.

---

## 6. Migrations-Strategie (inkrementell, 5 Phasen)

Jede Phase ist ein eigener PR, jeweils grün & mergebar.

### Phase 1: Port-Extraction (niedrigstes Risiko)
- Neues Verzeichnis `src/ports/`
- `ComputeBackend` Trait definieren
- Triviale `CpuBackend` Impl erstellen, die aktuelle Funktionen aufruft
- Noch keine Änderung an `cache/pqo.rs` etc.
- **Tests:** Backend-Trait-Tests mit Mock
- **LOC:** +200 / −0

### Phase 2: CUDA-Backend-Adapter (GPU-Dispatch bündeln)
- `src/adapters/backend_cuda/` mit `tensor_ptr.rs` Helper
- `CudaBackend` impl `ComputeBackend` — delegiert an bestehende `cache/cuda/`-Funktionen
- Noch keine Änderung an PqoCache
- **Tests:** 84-Zeilen-Unpack Dedup verifizierbar
- **LOC:** +400 / −0 (old code stays, new adapter wraps)

### Phase 3: Bridges aufräumen (Kernschnitt)
- `PqoCache` und `TqCache` auf `ComputeBackend` umstellen
- Alle `#[cfg(feature = "cuda")]` aus `cache/pqo.rs`, `cache/tq.rs`, `cache/quantize_tensor.rs` entfernen
- **Tests:** Existierende cache_tests müssen unverändert passen (Regression-Gate!)
- **LOC:** −300 / +50 (meist Löschen)

### Phase 4: Domain-Umorganisation
- `attention.rs` (1646) und `qjl.rs` (1487) zerlegen nach Zielverzeichnis
- `codebook/`, `rotation/`, `packing/`, `qjl/` in `domain/` umziehen
- `GpuPrecomputed` → `PrecomputeBuilder`
- **Tests:** Reines Re-Strukturieren, keine Logik-Änderung — `cargo test` muss alle ~364 Tests grün lassen
- **LOC:** Verschieben, keine Netto-Änderung

### Phase 5: Port-Bereinigung + Doku
- Optionalen `TensorView`-Port einführen (falls Metal/Vulkan wirklich kommt)
- README und docs/ aktualisieren
- Architektur-Regel-Test: `cargo-architecture` / `cargo-depgraph` CI-Check, dass lower→upper nicht passiert
- **LOC:** +100 docs

Gesamtaufwand geschätzt: **~3-5 Arbeitstage senior-level**, verteilbar über mehrere Wochen.

---

## 7. Nicht-Ziele & bewusste Trade-offs

- **Kein `TensorView`-Port im ersten Schritt.** Candle bleibt der einzige Tensor-Backend. Erst abstrahieren wenn Metal/Vulkan-Impl konkret geplant ist.
- **Keine async/tokio-Umstellung.** Quantisierung ist sync, gehört so.
- **Keine Dependency-Injection-Framework** (`shaku`, `di` o.ä.). Trait-Objects + Constructor-Injection reichen.
- **Kein Wechsel der Error-Strategie.** `thiserror` + `TurboQuantError` bleiben.
- **Kein Wegbrechen der Public API.** `lib.rs` re-exportiert weiterhin `PqoCache`, `TqCache`, `TurboQuantConfig` usw. — mistral.rs-Consumer müssen nichts ändern.
- **Kein Rewrite des QJL-Algorithmus.** Struktur ändern, Semantik nicht.

---

## 8. Offene Fragen

1. **Wird Metal/Vulkan mittelfristig (≤6 Monate) kommen?** Falls ja, `TensorView`-Port jetzt mit einziehen. Falls nein, YAGNI.
2. **Soll `mistralrs-kv-cache` (separate Crate) aufgelöst werden?** Die Trait-Definitionen könnten als `ports::cache` in turboquant leben. Momentan ist die Trennung etwas künstlich.
3. **Architekturregel-Checks in CI?** `cargo-architecture` bietet TOML-basierte Dependency-Regeln. Bei einem Proposal wie diesem wertvoll.
4. **Benchmarks als Regression-Gate.** Nach Phase 3: Vorher/Nachher-Benchmark im CI, damit kein Dispatch-Overhead durch die neue Abstraktion entsteht.

---

## 9. Erfolgs-Metriken

Nach Abschluss aller Phasen:

| Metrik | Vorher | Ziel |
|---|---|---|
| Zeilen in `cache/pqo.rs` | 236 | <100 |
| Zeilen in `attention.rs` | 1646 | 0 (aufgelöst) |
| Zeilen in `qjl.rs` | 1487 | <200 (reine Projektion) |
| `#[cfg(feature = "cuda")]` im Code | 9+ Stellen | ≤3 (nur Adapter-Gates) |
| Duplizierter Pointer-Extract | 2 × 84 Zeilen | 0 (via `CudaPtr<T>`) |
| Öffentliche Trait-Abstraktionen | 1 (`CompressedKVCache`) | 2 (+ `ComputeBackend`) |
| Tests, die ohne cuda-Feature laufen | ~70% | >90% |
| Public-API-Brüche | — | 0 |
