# Rustqual bug reports

Issues observed in rustqual 0.5.6 while bringing turboquant's `src/` to
Quality Score 100%. Each section is a self-contained bug report; copy the
relevant block into a rustqual issue.

---

## Bug 1: TQ_UNTESTED false-positive for functions tested from integration tests

**Rustqual version:** 0.5.6
**Check:** `TQ_UNTESTED`

### Summary

A `#[doc(hidden)] pub fn` declared in `src/cache/mod.rs` and directly
exercised by an integration test in `tests/` is still flagged
`TQ_UNTESTED`, even when the test function has the exact same name as the
production function and calls it with a fully-qualified path.

### Minimal reproduction

**`src/cache/mod.rs`** (relevant excerpt):
```rust
use std::sync::OnceLock;
use candle_core::{Device, Result};

pub use precomputed::GpuPrecomputed;
pub use config::CacheConfig;

#[doc(hidden)]
pub fn ensure_gpu_precomputed<'a>(
    cell: &'a OnceLock<GpuPrecomputed>,
    config: &CacheConfig,
    device: &Device,
) -> Result<&'a GpuPrecomputed> {
    if let Some(p) = cell.get() {
        return Ok(p);
    }
    let fresh = GpuPrecomputed::new(config, device)?;
    let _ = cell.set(fresh);
    match cell.get() {
        Some(p) => Ok(p),
        None => Err(candle_core::Error::Msg("init race".into())),
    }
}
```

**`tests/cache_internals_tests.rs`**:
```rust
#![cfg(feature = "candle")]

use std::sync::OnceLock;
use candle_core::Device;
use turboquant::cache::config::QuantNormMode;
use turboquant::cache::{CacheConfig, GpuPrecomputed};

#[test]
fn ensure_gpu_precomputed() {
    let cell: OnceLock<GpuPrecomputed> = OnceLock::new();
    let cfg = CacheConfig {
        bits: 3,
        head_dim: 128,
        num_kv_heads: 4,
        num_layers: 2,
        norm_mode: QuantNormMode::MaxNorm,
        outlier_blocks: usize::MAX,
    };
    let p1 = turboquant::cache::ensure_gpu_precomputed(&cell, &cfg, &Device::Cpu).unwrap();
    let p2 = turboquant::cache::ensure_gpu_precomputed(&cell, &cfg, &Device::Cpu).unwrap();
    assert_eq!(p1 as *const _, p2 as *const _);
}

#[test]
fn ensure_gpu_precomputed_returns_initialized_cell() {
    let cell: OnceLock<GpuPrecomputed> = OnceLock::new();
    let cfg = /* as above */;
    assert!(cell.get().is_none());
    let _ = turboquant::cache::ensure_gpu_precomputed(&cell, &cfg, &Device::Cpu).unwrap();
    assert!(cell.get().is_some());
}
```

### Observed behaviour

```
$ cargo test --features candle --test cache_internals_tests
running 2 tests
test ensure_gpu_precomputed ... ok
test ensure_gpu_precomputed_returns_initialized_cell ... ok

$ rustqual src/
  cache/mod.rs:46  TQ_UNTESTED  in ensure_gpu_precomputed
```

### Expected behaviour

The function should be considered tested. Both test function names share
the production function's name as prefix (or are exactly equal in the first
case), and both tests call the function directly through the public path.

### Observations

- The finding persists regardless of:
  - Where the function is declared (`src/cache/mod.rs`, `src/cache/common.rs`)
  - Whether it is `pub` directly, `pub(crate)` + `pub use` re-export, or
    `#[doc(hidden)] pub`
  - Whether the test function name exactly matches (`ensure_gpu_precomputed`)
    or is a prefix-extended name
    (`ensure_gpu_precomputed_returns_initialized_cell`)
- Other public `impl` methods tested via the same cross-file pattern (e.g.
  `LayerStorage::seq_len` called from `cache_storage_tests.rs::layer_storage_default_is_empty`)
  show the same TQ_UNTESTED behaviour and require `// qual:allow(TQ-003)`.

### Root cause hypothesis

Rustqual's TQ_UNTESTED heuristic appears not to trace cross-crate
integration test calls to free functions (or at least not when declared at
module root). The check seems to rely on something other than name matching
or import-graph reachability.

### Workaround

Suppress with `// qual:allow(TQ-003) — tested via <test-file>`. But see
Bug 3 below regarding the qual:allow comment parser.

---

## Bug 2: SRP_STRUCT LCOM4=2 false-positive on data-layer struct with paired reader/mutator methods

**Rustqual version:** 0.5.6
**Check:** `SRP_STRUCT` / `LCOM4`

### Summary

A simple data-holder struct with both `&self` readers and `&mut self`
mutators that all touch overlapping fields (including a `validate()` method
that explicitly reads every field, and a mutator that calls `validate()`
via `debug_assert!`) still computes LCOM4=2. Rustqual appears to partition
methods by `&self` vs. `&mut self` receiver kind rather than by field
access sets.

### Minimal reproduction

```rust
pub struct LayerStorage {
    buf_seq_len: usize,
    gpu_k_indices: Option<Tensor>,
    gpu_v_indices: Option<Tensor>,
    gpu_k_scales: Option<Tensor>,
    gpu_v_scales: Option<Tensor>,
    gpu_path_active: bool,
}

impl LayerStorage {
    // &self readers
    pub fn seq_len(&self) -> usize { self.buf_seq_len }
    pub fn is_active(&self) -> bool { self.gpu_path_active && self.buf_seq_len > 0 }
    pub fn capacity(&self) -> usize { self.gpu_k_indices.as_ref().map_or(0, |t| t.dims()[1]) }
    pub fn buffers(&self) -> Option<LayerBuffers<'_>> {
        // touches all four gpu_* fields
    }
    pub fn memory_usage(&self, metadata: &StorageMetadata) -> usize {
        if self.buf_seq_len == 0 { return 0; } /* ... */
    }
    pub fn validate(&self) -> Result<()> {
        // reads buf_seq_len, gpu_path_active, all four gpu_* fields
        if self.gpu_path_active && self.buf_seq_len == 0 { /* ... */ }
        if self.gpu_path_active {
            if self.gpu_k_indices.is_none() || self.gpu_v_indices.is_none() { /* ... */ }
            if self.gpu_k_scales.is_none() || self.gpu_v_scales.is_none() { /* ... */ }
        }
        Ok(())
    }

    // &mut self mutators
    pub fn ensure_capacity(&mut self, /* ... */) -> Result<()> {
        // writes all gpu_* fields, reads buf_seq_len
    }
    pub fn append(&mut self, /* ... */) -> Result<()> {
        // writes buf_seq_len, gpu_path_active, reads gpu_* fields
        self.buf_seq_len = offset + new_seq_len;
        self.gpu_path_active = true;
        debug_assert!(self.validate().is_ok());  // ← calls reader
        Ok(())
    }
    pub fn reset(&mut self) { *self = Self::default(); }
}
```

### Observed behaviour

```
$ rustqual src/
  cache/storage.rs:52  SRP_STRUCT  LCOM4=2  in LayerStorage
```

### Expected behaviour

LCOM4 should compute 1, because:
1. Every reader and mutator touches `buf_seq_len` (directly or indirectly).
2. `validate()` reads every field of the struct.
3. `append()` calls `validate()`, establishing a direct method-call link
   between the mutator and reader sets.

### Observations

- Adding a catch-all reader method (`validate()` touching all fields) did
  not reduce LCOM4 from 2 to 1.
- Calling the reader from a mutator via `debug_assert!(self.validate())`
  also did not reduce LCOM4.
- Rustqual's reported "2 clusters" appears to partition the method set
  into `{&self readers}` and `{&mut self mutators}` regardless of shared
  field accesses or call edges between them.

### Root cause hypothesis

LCOM4 is being computed on a method graph where `&self` and `&mut self`
methods are in disjoint components, even when they share all field
accesses. In classic LCOM4 (Hitz & Montazeri), two methods are connected
if they access a common attribute OR one calls the other; the Rust receiver
kind should not affect connectedness.

### Workaround

Suppress with `// qual:allow(srp) — <reason>` immediately above the struct
declaration (must stay on a single line; see Bug 3).

---

## Bug 3: `qual:allow` annotation only honoured on a single comment line

**Rustqual version:** 0.5.6

### Summary

When a `// qual:allow(TOKEN)` annotation is followed by additional
explanatory `//` comment lines on subsequent rows before the decorated
item, the suppression is silently dropped. Only a single-line `qual:allow`
comment (with optional em-dash continuation on the same line) is honoured.

### Minimal reproduction

**This is silently ignored:**
```rust
// qual:allow(srp) — rustqual false-positive LCOM4=2.
// The struct's methods form one coherent data-layer abstraction.
// See docs/rustqual-bugs.md for details.
#[derive(Default)]
pub struct LayerStorage { /* ... */ }
```

**This is honoured:**
```rust
// qual:allow(srp) — cohesive per-layer GPU storage: readers and mutators
#[derive(Default)]
pub struct LayerStorage { /* ... */ }
```

### Observed behaviour

Only the single-line form reduces the finding count. The multi-line form
reports the SRP_STRUCT finding as if no suppression existed.

### Expected behaviour

Either (a) all consecutive `//` comments immediately above the item should
be treated as part of the suppression rationale, or (b) rustqual should
warn at parse time when a `qual:allow` comment is not the single line
immediately preceding the item.

### Observations

- Placement rules that do seem to work:
  - Comment immediately above the `pub fn` / `pub struct` (no
    `#[derive(...)]` in between, no additional `//` lines in between).
  - Doc comments (`///`) above the suppression are fine; only additional
    `//` line comments break it.
- `#[derive(...)]` between the annotation and the item also breaks it for
  structs, though in some files a `#[derive(...)]` followed immediately by
  the `// qual:allow(...)` on the very next line works.

### Impact

Silent loss of suppression is particularly painful because the finding
reappears without any indication that the intended `qual:allow` was
malformed. A warning ("`qual:allow` comment found but not directly
preceding an item") would make this debuggable.

---

## Environment

- `rustqual 0.5.6`
- `cargo 1.X` (stable toolchain)
- Project: turboquant-rs, `candle` feature enabled
- Reproducing repository: https://github.com/SaschaOnTour/turboquant
  (branch with the refactor: ask Sascha for the exact SHA)
