use thiserror::Error;

/// Errors that can occur during TurboQuant operations.
#[derive(Debug, Error)]
pub enum TurboQuantError {
    /// The vector dimension is not a power of two, which is required for WHT.
    #[error("dimension {0} is not a power of two")]
    InvalidDimension(usize),

    /// The bit width is not supported (must be 3 or 4).
    #[error("unsupported bit width {0}, expected 3 or 4")]
    UnsupportedBits(u8),

    /// The input vector length does not match the configured dimension.
    #[error("expected vector of length {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// A layer index is out of range.
    #[error("layer index {index} out of range (num_layers = {num_layers})")]
    LayerOutOfRange { index: usize, num_layers: usize },
}

/// Result type alias for TurboQuant operations.
pub type Result<T> = std::result::Result<T, TurboQuantError>;
