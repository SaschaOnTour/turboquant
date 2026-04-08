//! Configuration types and constants for the compressed KV-cache.

/// Block size for quantization. Each head_dim vector is split into blocks
/// of this size, each independently quantized with its own WHT rotation and norm.
///
/// Paper Section 4.3 + llama.cpp: block_size=32 gives much better quality
/// for real LLM KV cache vectors (which have norms 10-400+) compared to
/// quantizing the full head_dim=128 as a single block.
pub const QUANT_BLOCK_SIZE: usize = 32;

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

impl std::fmt::Display for QuantNormMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MaxNorm => write!(f, "maxnorm"),
            Self::L2Norm => write!(f, "l2norm"),
        }
    }
}
