use thiserror::Error;

/// Errors that can occur during TurboQuant operations.
#[derive(Debug, Error)]
pub enum TurboQuantError {
    /// The vector dimension is not a power of two, which is required for WHT.
    #[error("dimension {0} is not a power of two")]
    InvalidDimension(usize),

    /// The bit width is not supported (must be 2, 3, or 4).
    #[error("unsupported bit width {0}, expected 2, 3, or 4")]
    UnsupportedBits(u8),

    /// The input vector length does not match the configured dimension.
    #[error("expected vector of length {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// A layer index is out of range.
    #[error("layer index {index} out of range (num_layers = {num_layers})")]
    LayerOutOfRange { index: usize, num_layers: usize },

    /// A range is out of bounds for the entry count.
    #[error("range [{start}..{end}) out of bounds for entry_count = {entry_count}")]
    RangeOutOfBounds {
        start: usize,
        end: usize,
        entry_count: usize,
    },
}

/// Result type alias for TurboQuant operations.
pub type Result<T> = std::result::Result<T, TurboQuantError>;

/// Maps a boolean condition to a Result, returning the given error when false.
///
/// Pure Operation: if/else only, no project function calls.
pub fn require(condition: bool, error: TurboQuantError) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(error)
    }
}

// ---------------------------------------------------------------------------
// Shared validation primitives
// ---------------------------------------------------------------------------

/// Checks whether two values are equal.
///
/// Pure Operation: comparison only.
pub(crate) fn values_match(a: usize, b: usize) -> bool {
    a == b
}

/// Validates that `actual` matches `expected`, returning a
/// [`TurboQuantError::DimensionMismatch`] on failure.
///
/// Pure Integration: only calls `require` and `values_match`.
pub(crate) fn check_values_match(actual: usize, expected: usize) -> Result<()> {
    require(
        values_match(actual, expected),
        TurboQuantError::DimensionMismatch { expected, actual },
    )
}
