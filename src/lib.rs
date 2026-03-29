//! TurboQuant — KV-Cache quantization with zero accuracy loss.
//!
//! Implements Google's TurboQuant algorithm (Zandieh et al., ICLR 2026)
//! for compressing LLM key-value caches to 3–4 bits per value.

pub mod attention;
pub mod codebook;
pub mod error;
pub mod packed;
pub mod qjl;
pub mod quantize;
pub mod rotation;
