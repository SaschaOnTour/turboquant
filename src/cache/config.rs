//! Configuration types and constants for the compressed KV-cache.

/// Block size for quantization. Each head_dim vector is split into blocks
/// of this size, each independently quantized with its own WHT rotation and norm.
///
/// Paper Section 4.3 + llama.cpp: block_size=32 gives much better quality
/// for real LLM KV cache vectors (which have norms 10-400+) compared to
/// quantizing the full head_dim=128 as a single block.
pub const QUANT_BLOCK_SIZE: usize = 32;

/// Bits per byte — used for byte-level packing/unpacking calculations.
pub const BITS_PER_BYTE: usize = 8;

/// Default seed for the deterministic WHT rotation sign pattern.
pub const DEFAULT_ROTATION_SEED: u64 = 42;

/// Default seed for QJL Rademacher matrix.
pub const DEFAULT_QJL_SEED: u64 = 12345;

/// Normalization strategy for block-level PolarQuant.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum QuantNormMode {
    /// Paper Algorithm 1: L2-norm -> unit sphere -> Beta-distribution codebooks.
    /// Mathematically optimal MSE. Use with QJL for unbiased inner products.
    L2Norm,
    /// llama.cpp approach: max-abs-norm -> [-1,1] range -> empirical codebooks.
    /// Better for attention quality without QJL. No theoretical guarantees.
    #[default]
    MaxNorm,
}

impl std::str::FromStr for QuantNormMode {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "maxnorm" => Ok(Self::MaxNorm),
            "l2norm" => Ok(Self::L2Norm),
            other => Err(format!(
                "Unknown norm mode `{other}`. Options: maxnorm, l2norm"
            )),
        }
    }
}

/// Configuration for TurboQuant compressed KV-cache.
///
/// Shared between [`PqoCache`], [`TqCache`], and [`GpuPrecomputed`].
/// Use `outlier_blocks = usize::MAX` for PQO mode (recommended),
/// or `outlier_blocks = 0` for TQ mode (with QJL correction).
#[derive(Clone, Debug)]
pub struct CacheConfig {
    /// Total bit-width (3 or 4). Polar codebook uses `bits - 1`.
    pub bits: u8,
    /// Elements per attention head.
    pub head_dim: usize,
    /// Number of KV attention heads.
    pub num_kv_heads: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Normalization strategy for block-level PolarQuant.
    pub norm_mode: QuantNormMode,
    /// Number of outlier blocks per head_dim vector.
    /// `usize::MAX` = PQO (all outlier), `0` = TQ (standard codebook + QJL).
    pub outlier_blocks: usize,
}

impl CacheConfig {
    /// Whether QJL correction is needed (derived from `outlier_blocks`).
    ///
    /// QJL is used when no outlier blocks are present (TQ mode, `outlier_blocks == 0`).
    pub fn qjl_enabled(&self) -> bool {
        self.outlier_blocks == 0
    }

    /// Number of quantization blocks per head_dim vector.
    pub fn num_blocks(&self) -> usize {
        self.head_dim / QUANT_BLOCK_SIZE
    }

    /// Packed dimension: bytes per token for indices.
    pub fn packed_dim(&self) -> usize {
        self.head_dim * self.bits as usize / BITS_PER_BYTE
    }
}

impl std::fmt::Display for QuantNormMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MaxNorm => write!(f, "maxnorm"),
            Self::L2Norm => write!(f, "l2norm"),
        }
    }
}
