//! Packed data structures for quantized blocks.
//!
//! After quantization, indices are bit-packed into compact representations
//! to minimise memory usage. TQ2 uses 2 bits, TQ3 uses 3 bits per value
//! (3.5 bpw for block_size=32), TQ4 uses 4 bits per value (4.5 bpw for
//! block_size=32).

use half::f16;

use crate::error::{require, Result, TurboQuantError};

// ---------------------------------------------------------------------------
// Named constants (eliminates magic numbers)
// ---------------------------------------------------------------------------

/// Bits per value for TQ2 quantization.
pub(crate) const BITS_TQ2: u8 = 2;

/// Bits per value for TQ3 quantization.
pub(crate) const BITS_TQ3: u8 = 3;

/// Bits per value for TQ4 quantization.
pub(crate) const BITS_TQ4: u8 = 4;

/// Number of indices packed into one 2-bit group.
const PACK_2BIT_GROUP_SIZE: usize = 4;

/// Number of indices packed into one 3-bit group.
const PACK_3BIT_GROUP_SIZE: usize = 8;

/// Number of bytes produced by packing one 3-bit group.
const PACK_3BIT_BYTES: usize = 3;

/// Number of indices packed into one 4-bit group.
const PACK_4BIT_GROUP_SIZE: usize = 2;

/// Bit mask for 3-bit values (0b111).
const MASK_3BIT: u8 = 0x7;

/// Bit mask for 2-bit values (0b11).
const MASK_2BIT: u8 = 0x3;

/// Bit mask for 1-bit values (0b1).
const MASK_1BIT: u8 = 0x1;

/// Bit mask for 4-bit values (0b1111).
const MASK_4BIT: u8 = 0xF;

/// Shift amount for 3-bit boundaries.
const SHIFT_3: u32 = 3;

/// Shift amount for 4-bit boundaries.
const SHIFT_4: u32 = 4;

/// Shift amount for 5-bit boundaries.
const SHIFT_5: u32 = 5;

/// Shift amount for 6-bit boundaries.
const SHIFT_6: u32 = 6;

/// Shift amount for 7-bit boundaries.
const SHIFT_7: u32 = 7;

/// Shift amount for 1-bit boundaries.
const SHIFT_1: u32 = 1;

/// Shift amount for 2-bit boundaries.
const SHIFT_2: u32 = 2;

/// Size of the f16 scale field in bytes.
const SCALE_SIZE_BYTES: usize = 2;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for TurboQuant quantization.
#[derive(Clone, Copy)]
pub struct TurboQuantConfig {
    /// Bits per value (2, 3, or 4).
    pub(crate) bits: u8,
    /// Vector dimension (must be a power of two for WHT).
    pub(crate) dim: usize,
    /// Seed for the rotation matrix.
    pub(crate) rotation_seed: u64,
}

/// Check whether `bits` is a supported value (2, 3, or 4).
///
/// Pure Operation: contains only logic, no calls to other project functions.
pub(crate) fn is_valid_bits(bits: u8) -> bool {
    bits == BITS_TQ2 || bits == BITS_TQ3 || bits == BITS_TQ4
}

/// Check whether `dim` is a non-zero power of two.
///
/// Pure Operation: contains only logic, no calls to other project functions.
pub(crate) fn is_valid_dim(dim: usize) -> bool {
    dim > 0 && dim.is_power_of_two()
}

impl TurboQuantConfig {
    /// Create a new configuration after validating inputs.
    ///
    /// Returns an error when `bits` is not 2, 3, or 4, or `dim` is not a power
    /// of two.
    ///
    /// Pure Integration: only calls `require`, `is_valid_bits`, `is_valid_dim`.
    pub fn new(bits: u8, dim: usize) -> Result<Self> {
        require(is_valid_bits(bits), TurboQuantError::UnsupportedBits(bits))?;
        require(is_valid_dim(dim), TurboQuantError::InvalidDimension(dim))?;
        Ok(Self {
            bits,
            dim,
            rotation_seed: 0,
        })
    }

    /// Builder-style setter for the rotation seed.
    // qual:api — public builder API for downstream consumers
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rotation_seed = seed;
        self
    }
}

// ---------------------------------------------------------------------------
// Unified PackedBlock
// ---------------------------------------------------------------------------

/// A packed quantized block that stores a scale factor and bit-packed indices.
///
/// Replaces the former `BlockTQ2`, `BlockTQ3`, and `BlockTQ4` structs with a
/// single type that tracks its own bit width.
pub struct PackedBlock {
    /// Bit width used for packing (2, 3, or 4).
    bits: u8,
    /// Scaling factor (L2-norm of original vector).
    scale: f16,
    /// Packed indices (layout depends on `bits`).
    packed_indices: Vec<u8>,
}

impl PackedBlock {
    /// Create a new packed block from a scale and a slice of unpacked index values.
    ///
    /// The indices are bit-packed internally based on the specified `bits` width.
    ///
    /// Pure Integration: delegates packing to the bit-width-specific helper
    /// selected by the `pack` closure (IOSP lenient-mode closure pattern).
    pub fn new(bits: u8, scale: f16, indices: &[u8]) -> Self {
        let pack = |indices: &[u8]| -> Vec<u8> {
            match bits {
                BITS_TQ2 => pack_indices_2bit(indices),
                BITS_TQ3 => pack_indices_3bit(indices),
                BITS_TQ4 => pack_indices_4bit(indices),
                _ => unreachable!("bits validated to be 2, 3, or 4"),
            }
        };
        Self {
            bits,
            scale,
            packed_indices: pack(indices),
        }
    }

    /// Returns the f16 scale factor stored in the block.
    ///
    /// Pure Operation: field access only (TRIVIAL classification).
    pub fn scale(&self) -> f16 {
        self.scale
    }

    /// Returns the bit width used for packing.
    ///
    /// Pure Operation: field access only (TRIVIAL classification).
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// Total size of the block in bytes (2 bytes for f16 scale + packed data).
    pub fn size_bytes(&self) -> usize {
        SCALE_SIZE_BYTES + self.packed_indices.len()
    }

    /// Unpacks stored indices into a caller-provided buffer, avoiding allocation.
    ///
    /// This is the hot-path variant: reuses the buffer across repeated calls
    /// (e.g. inside attention score loops) to eliminate per-key allocations.
    ///
    /// Pure Integration: delegates unpacking to the bit-width-specific helper
    /// selected by the `do_unpack` closure (IOSP lenient-mode closure pattern).
    pub fn unpack_into(&self, count: usize, buf: &mut Vec<u8>) {
        buf.clear();
        let do_unpack = |packed: &[u8], out: &mut Vec<u8>| match self.bits {
            BITS_TQ2 => out.extend_from_slice(&unpack_indices_2bit(packed, count)),
            BITS_TQ3 => out.extend_from_slice(&unpack_indices_3bit(packed, count)),
            BITS_TQ4 => out.extend_from_slice(&unpack_indices_4bit(packed, count)),
            _ => unreachable!("bits validated"),
        };
        do_unpack(&self.packed_indices, buf);
        buf.truncate(count);
    }

    /// Recover the unpacked index values.
    ///
    /// Allocates a fresh buffer. For hot paths, prefer
    /// [`unpack_into`](Self::unpack_into) with a reusable buffer.
    pub fn unpack(&self, count: usize) -> Vec<u8> {
        let do_unpack = |packed: &[u8]| match self.bits {
            BITS_TQ2 => unpack_indices_2bit(packed, count),
            BITS_TQ3 => unpack_indices_3bit(packed, count),
            BITS_TQ4 => unpack_indices_4bit(packed, count),
            _ => unreachable!("bits validated"),
        };
        do_unpack(&self.packed_indices)
    }
}

// ---------------------------------------------------------------------------
// 2-bit packing / unpacking  (pure Operation functions)
// ---------------------------------------------------------------------------

/// Pack 4 two-bit values into 1 byte.
///
/// Only the lowest 2 bits of each input byte are used.
pub fn pack_2bit(values: &[u8; PACK_2BIT_GROUP_SIZE]) -> u8 {
    (values[0] & MASK_2BIT)
        | ((values[1] & MASK_2BIT) << SHIFT_2)
        | ((values[2] & MASK_2BIT) << SHIFT_4)
        | ((values[3] & MASK_2BIT) << SHIFT_6)
}

/// Unpack 1 byte into 4 two-bit values.
pub fn unpack_2bit(packed: u8) -> [u8; PACK_2BIT_GROUP_SIZE] {
    [
        packed & MASK_2BIT,
        (packed >> SHIFT_2) & MASK_2BIT,
        (packed >> SHIFT_4) & MASK_2BIT,
        (packed >> SHIFT_6) & MASK_2BIT,
    ]
}

// ---------------------------------------------------------------------------
// 2-bit vector-level helpers  (pure Operation -- arithmetic/logic only)
// ---------------------------------------------------------------------------

/// Compute the number of full groups of 4 that fit in `len` elements.
///
/// Pure Operation: arithmetic only.
fn num_2bit_groups(len: usize) -> usize {
    len / PACK_2BIT_GROUP_SIZE
}

/// Check whether `len` elements have a remainder after grouping by 4.
///
/// Pure Operation: arithmetic only.
fn has_2bit_remainder(len: usize) -> bool {
    len % PACK_2BIT_GROUP_SIZE != 0
}

/// Compute capacity for the packed 2-bit byte vector.
///
/// Pure Operation: arithmetic only.
fn packed_2bit_capacity(num_groups: usize, has_remainder: bool) -> usize {
    num_groups + usize::from(has_remainder)
}

/// Convert a chunk of exactly 4 bytes into the fixed-size array expected by
/// `pack_2bit`.
///
/// Pure Operation: slice-to-array conversion only.
fn chunk_to_2bit_array(chunk: &[u8]) -> [u8; PACK_2BIT_GROUP_SIZE] {
    chunk.try_into().expect("chunk size matches group size")
}

/// Pad a remainder slice (< 4 elements) into a full 4-element array, filling
/// the tail with zeros.
///
/// Pure Operation: copy only.
fn pad_remainder_2bit(tail: &[u8]) -> [u8; PACK_2BIT_GROUP_SIZE] {
    let mut padded = [0u8; PACK_2BIT_GROUP_SIZE];
    padded[..tail.len()].copy_from_slice(tail);
    padded
}

// ---------------------------------------------------------------------------
// 2-bit vector-level packing  (pure Integration functions)
// ---------------------------------------------------------------------------

/// Pack a full vector of 2-bit indices into a compact byte vector.
///
/// Each group of 4 indices produces 1 byte. If the length is not a multiple of
/// 4, the remainder is zero-padded.
///
/// Pure Integration: delegates to `pack_indices_chunked`.
pub fn pack_indices_2bit(indices: &[u8]) -> Vec<u8> {
    pack_indices_chunked(
        indices,
        PACK_2BIT_GROUP_SIZE,
        packed_2bit_capacity(
            num_2bit_groups(indices.len()),
            has_2bit_remainder(indices.len()),
        ),
        |chunk, out| out.push(pack_2bit(&chunk_to_2bit_array(chunk))),
        |tail, out| out.push(pack_2bit(&pad_remainder_2bit(tail))),
    )
}

/// Unpack a compact byte vector into `count` 2-bit index values.
///
/// Pure Integration: orchestrates helpers and `unpack_2bit`, no inline logic.
pub fn unpack_indices_2bit(packed: &[u8], count: usize) -> Vec<u8> {
    let mut result = Vec::with_capacity(count);

    for &byte in packed {
        let vals = unpack_2bit(byte);
        result.extend_from_slice(&vals);
    }

    result.truncate(count);
    result
}

// ---------------------------------------------------------------------------
// 3-bit packing / unpacking  (pure Operation functions)
// ---------------------------------------------------------------------------

/// Pack 8 three-bit values into 3 bytes.
///
/// Only the lowest 3 bits of each input byte are used.
pub fn pack_3bit(values: &[u8; PACK_3BIT_GROUP_SIZE]) -> [u8; PACK_3BIT_BYTES] {
    let mut packed = [0u8; PACK_3BIT_BYTES];
    packed[0] = (values[0] & MASK_3BIT)
        | ((values[1] & MASK_3BIT) << SHIFT_3)
        | ((values[2] & MASK_2BIT) << SHIFT_6);
    packed[1] = ((values[2] >> SHIFT_2) & MASK_1BIT)
        | ((values[3] & MASK_3BIT) << SHIFT_1)
        | ((values[4] & MASK_3BIT) << SHIFT_4)
        | ((values[5] & MASK_1BIT) << SHIFT_7);
    packed[2] = ((values[5] >> SHIFT_1) & MASK_2BIT)
        | ((values[6] & MASK_3BIT) << SHIFT_2)
        | ((values[7] & MASK_3BIT) << SHIFT_5);
    packed
}

/// Unpack 3 bytes into 8 three-bit values.
pub fn unpack_3bit(packed: &[u8; PACK_3BIT_BYTES]) -> [u8; PACK_3BIT_GROUP_SIZE] {
    let mut values = [0u8; PACK_3BIT_GROUP_SIZE];
    values[0] = packed[0] & MASK_3BIT;
    values[1] = (packed[0] >> SHIFT_3) & MASK_3BIT;
    values[2] = ((packed[0] >> SHIFT_6) & MASK_2BIT) | ((packed[1] & MASK_1BIT) << SHIFT_2);
    values[3] = (packed[1] >> SHIFT_1) & MASK_3BIT;
    values[4] = (packed[1] >> SHIFT_4) & MASK_3BIT;
    values[5] = ((packed[1] >> SHIFT_7) & MASK_1BIT) | ((packed[2] & MASK_2BIT) << SHIFT_1);
    values[6] = (packed[2] >> SHIFT_2) & MASK_3BIT;
    values[7] = (packed[2] >> SHIFT_5) & MASK_3BIT;
    values
}

// ---------------------------------------------------------------------------
// 4-bit packing / unpacking  (pure Operation functions)
// ---------------------------------------------------------------------------

/// Pack 2 four-bit values into 1 byte.
///
/// Only the lowest 4 bits of each input byte are used.
pub fn pack_4bit(values: &[u8; 2]) -> u8 {
    (values[0] & MASK_4BIT) | ((values[1] & MASK_4BIT) << SHIFT_4)
}

/// Unpack 1 byte into 2 four-bit values.
pub fn unpack_4bit(packed: u8) -> [u8; 2] {
    [packed & MASK_4BIT, (packed >> SHIFT_4) & MASK_4BIT]
}

// ---------------------------------------------------------------------------
// 3-bit vector-level helpers  (pure Operation -- arithmetic/logic only)
// ---------------------------------------------------------------------------

/// Compute the number of full groups of 8 that fit in `len` elements.
///
/// Pure Operation: arithmetic only.
fn num_3bit_groups(len: usize) -> usize {
    len / PACK_3BIT_GROUP_SIZE
}

/// Check whether `len` elements have a remainder after grouping by 8.
///
/// Pure Operation: arithmetic only.
fn has_3bit_remainder(len: usize) -> bool {
    len % PACK_3BIT_GROUP_SIZE != 0
}

/// Compute capacity for the packed 3-bit byte vector.
///
/// Pure Operation: arithmetic only, no calls to other project functions.
fn packed_3bit_capacity(num_groups: usize, has_remainder: bool) -> usize {
    let remainder_bytes = if has_remainder { PACK_3BIT_BYTES } else { 0 };
    num_groups * PACK_3BIT_BYTES + remainder_bytes
}

/// Convert a chunk of exactly 8 bytes into the fixed-size array expected by
/// `pack_3bit`.
///
/// Pure Operation: slice-to-array conversion only.
fn chunk_to_3bit_array(chunk: &[u8]) -> [u8; PACK_3BIT_GROUP_SIZE] {
    chunk.try_into().expect("chunk size matches group size")
}

/// Pad a remainder slice (< 8 elements) into a full 8-element array, filling
/// the tail with zeros.
///
/// Pure Operation: copy only.
fn pad_remainder_3bit(tail: &[u8]) -> [u8; PACK_3BIT_GROUP_SIZE] {
    let mut padded = [0u8; PACK_3BIT_GROUP_SIZE];
    padded[..tail.len()].copy_from_slice(tail);
    padded
}

/// Convert a 3-byte chunk into the fixed-size array expected by `unpack_3bit`.
///
/// Pure Operation: slice-to-array conversion only.
fn chunk_to_packed_3bit_array(chunk: &[u8]) -> [u8; PACK_3BIT_BYTES] {
    chunk.try_into().expect("chunk size matches group size")
}

// ---------------------------------------------------------------------------
// 4-bit vector-level helpers  (pure Operation -- arithmetic/logic only)
// ---------------------------------------------------------------------------

/// Compute the number of full pairs that fit in `len` elements.
///
/// Pure Operation: arithmetic only.
fn num_4bit_pairs(len: usize) -> usize {
    len / PACK_4BIT_GROUP_SIZE
}

/// Check whether `len` elements have a trailing odd element.
///
/// Pure Operation: arithmetic only.
fn has_4bit_remainder(len: usize) -> bool {
    len % PACK_4BIT_GROUP_SIZE != 0
}

/// Compute capacity for the packed 4-bit byte vector.
///
/// Pure Operation: arithmetic only.
fn packed_4bit_capacity(num_pairs: usize, has_remainder: bool) -> usize {
    num_pairs + usize::from(has_remainder)
}

/// Convert a 2-byte chunk into the fixed-size array expected by `pack_4bit`.
///
/// Pure Operation: slice-to-array conversion only.
fn chunk_to_4bit_array(pair: &[u8]) -> [u8; PACK_4BIT_GROUP_SIZE] {
    pair.try_into().expect("chunk size matches group size")
}

/// Build the pair for packing a trailing odd element (high nibble is zero).
///
/// Pure Operation: value construction only.
fn trailing_4bit_pair(last: u8) -> [u8; PACK_4BIT_GROUP_SIZE] {
    [last, 0]
}

// ---------------------------------------------------------------------------
// Generic chunked packing helper  (pure Integration)
// ---------------------------------------------------------------------------

/// Generic bit-packing: splits `indices` into chunks of `group_size`, packs
/// each chunk with `pack_group`, and handles the remainder with `pack_remainder`.
///
/// Pure Integration: only calls the provided closures and extends the output vector.
fn pack_indices_chunked<F, R>(
    indices: &[u8],
    group_size: usize,
    capacity: usize,
    mut pack_group: F,
    mut pack_remainder: R,
) -> Vec<u8>
where
    F: FnMut(&[u8], &mut Vec<u8>),
    R: FnMut(&[u8], &mut Vec<u8>),
{
    let mut packed = Vec::with_capacity(capacity);
    for chunk in indices.chunks_exact(group_size) {
        pack_group(chunk, &mut packed);
    }
    let mut handle_tail = || {
        let tail = indices.chunks_exact(group_size).remainder();
        if tail.is_empty() {
            return;
        }
        pack_remainder(tail, &mut packed);
    };
    handle_tail();
    packed
}

// ---------------------------------------------------------------------------
// Vector-level packing  (pure Integration functions -- delegate only)
// ---------------------------------------------------------------------------

/// Pack a full vector of 3-bit indices into a compact byte vector.
///
/// The input length must be a multiple of 8. Each group of 8 indices produces 3
/// bytes.
///
/// Pure Integration: delegates to `pack_indices_chunked`.
pub fn pack_indices_3bit(indices: &[u8]) -> Vec<u8> {
    pack_indices_chunked(
        indices,
        PACK_3BIT_GROUP_SIZE,
        packed_3bit_capacity(
            num_3bit_groups(indices.len()),
            has_3bit_remainder(indices.len()),
        ),
        |chunk, out| out.extend_from_slice(&pack_3bit(&chunk_to_3bit_array(chunk))),
        |tail, out| out.extend_from_slice(&pack_3bit(&pad_remainder_3bit(tail))),
    )
}

/// Unpack a compact byte vector into `count` 3-bit index values.
///
/// Pure Integration: orchestrates helpers and `unpack_3bit`, no inline logic.
pub fn unpack_indices_3bit(packed: &[u8], count: usize) -> Vec<u8> {
    let mut result = Vec::with_capacity(count);

    for chunk in packed.chunks_exact(PACK_3BIT_BYTES) {
        let arr = chunk_to_packed_3bit_array(chunk);
        let vals = unpack_3bit(&arr);
        result.extend_from_slice(&vals);
    }

    result.truncate(count);
    result
}

/// Pack a full vector of 4-bit indices into a compact byte vector.
///
/// Each pair of indices produces 1 byte. If the length is odd the last index is
/// packed alone (high nibble is zero).
///
/// Pure Integration: delegates to `pack_indices_chunked`.
pub fn pack_indices_4bit(indices: &[u8]) -> Vec<u8> {
    pack_indices_chunked(
        indices,
        PACK_4BIT_GROUP_SIZE,
        packed_4bit_capacity(
            num_4bit_pairs(indices.len()),
            has_4bit_remainder(indices.len()),
        ),
        |chunk, out| out.push(pack_4bit(&chunk_to_4bit_array(chunk))),
        |tail, out| out.push(pack_4bit(&trailing_4bit_pair(tail[0]))),
    )
}

/// Unpack a compact byte vector into `count` 4-bit index values.
pub fn unpack_indices_4bit(packed: &[u8], count: usize) -> Vec<u8> {
    let mut result = Vec::with_capacity(count);

    for &byte in packed {
        let vals = unpack_4bit(byte);
        result.extend_from_slice(&vals);
    }

    result.truncate(count);
    result
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Standard block size (power of two) used in config validation tests.
    const TEST_BLOCK_SIZE: usize = 32;
    /// Standard large dimension (power of two) used in config validation tests.
    const TEST_DIM_128: usize = 128;
    /// Number of 3-bit groups in capacity tests.
    const TEST_3BIT_GROUPS: usize = 4;
    /// Number of 4-bit pairs in capacity tests.
    const TEST_4BIT_PAIRS: usize = 5;
    /// Maximum valid 3-bit value (2^3 - 1).
    const MAX_3BIT_VALUE: u8 = 7;
    /// Maximum valid 4-bit value (2^4 - 1).
    const MAX_4BIT_VALUE: u8 = 15;
    /// Test trailing-pair input value.
    const TEST_TRAILING_VALUE: u8 = 9;
    /// Number of 3-bit indices in exact-multiple roundtrip test (2 groups of 8).
    const TEST_3BIT_EXACT_COUNT: usize = 16;
    /// Number of 3-bit indices in remainder roundtrip test.
    const TEST_3BIT_REMAINDER_COUNT: usize = 11;
    /// Number of 4-bit indices in even-count roundtrip test.
    const TEST_4BIT_EVEN_COUNT: usize = 10;
    /// Number of 4-bit indices in odd-count roundtrip test.
    const TEST_4BIT_ODD_COUNT: usize = 7;
    /// Number of 4-bit levels (2^4).
    const TEST_4BIT_LEVELS: u8 = 16;

    /// Maximum valid 2-bit value (2^2 - 1).
    const MAX_2BIT_VALUE: u8 = 3;
    /// Number of 2-bit indices in exact-multiple roundtrip test (3 groups of 4).
    const TEST_2BIT_EXACT_COUNT: usize = 12;
    /// Number of 2-bit indices in remainder roundtrip test.
    const TEST_2BIT_REMAINDER_COUNT: usize = 7;

    // -- is_valid_bits -------------------------------------------------------

    #[test]
    fn is_valid_bits_accepts_2_3_and_4() {
        assert!(is_valid_bits(BITS_TQ2));
        assert!(is_valid_bits(BITS_TQ3));
        assert!(is_valid_bits(BITS_TQ4));
    }

    #[test]
    fn is_valid_bits_rejects_others() {
        assert!(!is_valid_bits(0));
        assert!(!is_valid_bits(1));
        assert!(!is_valid_bits(5));
    }

    // -- is_valid_dim --------------------------------------------------------

    #[test]
    fn is_valid_dim_accepts_powers_of_two() {
        assert!(is_valid_dim(TEST_DIM_128 / 2));
        assert!(is_valid_dim(TEST_DIM_128));
    }

    #[test]
    fn is_valid_dim_rejects_invalid() {
        assert!(!is_valid_dim(0));
        assert!(!is_valid_dim(3));
        assert!(!is_valid_dim(100));
    }

    // -- packed_3bit_capacity ------------------------------------------------

    #[test]
    fn packed_3bit_capacity_no_remainder() {
        // 4 groups of 8 -> 4 * 3 = 12 bytes
        assert_eq!(
            packed_3bit_capacity(TEST_3BIT_GROUPS, false),
            TEST_3BIT_GROUPS * PACK_3BIT_BYTES
        );
    }

    #[test]
    fn packed_3bit_capacity_with_remainder() {
        // 4 groups + remainder -> 4 * 3 + 3 = 15 bytes
        assert_eq!(
            packed_3bit_capacity(TEST_3BIT_GROUPS, true),
            TEST_3BIT_GROUPS * PACK_3BIT_BYTES + PACK_3BIT_BYTES
        );
    }

    #[test]
    fn packed_3bit_capacity_zero_groups() {
        assert_eq!(packed_3bit_capacity(0, false), 0);
        assert_eq!(packed_3bit_capacity(0, true), 3);
    }

    // -- packed_4bit_capacity ------------------------------------------------

    #[test]
    fn packed_4bit_capacity_no_remainder() {
        assert_eq!(
            packed_4bit_capacity(TEST_4BIT_PAIRS, false),
            TEST_4BIT_PAIRS
        );
    }

    #[test]
    fn packed_4bit_capacity_with_remainder() {
        assert_eq!(
            packed_4bit_capacity(TEST_4BIT_PAIRS, true),
            TEST_4BIT_PAIRS + 1
        );
    }

    // -- chunk_to_3bit_array / chunk_to_4bit_array ---------------------------

    #[test]
    fn chunk_to_3bit_array_preserves_values() {
        let input: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let arr = chunk_to_3bit_array(&input);
        assert_eq!(arr, [0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn chunk_to_4bit_array_preserves_values() {
        let input: Vec<u8> = vec![10, 15];
        let arr = chunk_to_4bit_array(&input);
        assert_eq!(arr, [10, 15]);
    }

    // -- pad_remainder_3bit --------------------------------------------------

    #[test]
    fn pad_remainder_3bit_pads_correctly() {
        let tail: Vec<u8> = vec![1, 2, 3];
        let padded = pad_remainder_3bit(&tail);
        assert_eq!(padded, [1, 2, 3, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn pad_remainder_3bit_single_element() {
        let tail: Vec<u8> = vec![5];
        let padded = pad_remainder_3bit(&tail);
        assert_eq!(padded, [5, 0, 0, 0, 0, 0, 0, 0]);
    }

    // -- trailing_4bit_pair --------------------------------------------------

    #[test]
    fn trailing_4bit_pair_handles_single_element() {
        let pair = trailing_4bit_pair(TEST_TRAILING_VALUE);
        assert_eq!(pair, [TEST_TRAILING_VALUE, 0]);
    }

    // -- chunk_to_packed_3bit_array ------------------------------------------

    #[test]
    fn chunk_to_packed_3bit_array_preserves_values() {
        let input: Vec<u8> = vec![0xAB, 0xCD, 0xEF];
        let arr = chunk_to_packed_3bit_array(&input);
        assert_eq!(arr, [0xAB, 0xCD, 0xEF]);
    }

    // -- 3-bit pack/unpack ---------------------------------------------------

    #[test]
    fn pack_unpack_3bit_identity() {
        let values: [u8; PACK_3BIT_GROUP_SIZE] = [0, 1, 2, 3, 4, 5, 6, MAX_3BIT_VALUE];
        let packed = pack_3bit(&values);
        let unpacked = unpack_3bit(&packed);
        assert_eq!(values, unpacked);
    }

    #[test]
    fn pack_unpack_3bit_zeros() {
        let values = [0u8; PACK_3BIT_GROUP_SIZE];
        assert_eq!(unpack_3bit(&pack_3bit(&values)), values);
    }

    #[test]
    fn pack_unpack_3bit_max() {
        let values = [MAX_3BIT_VALUE; PACK_3BIT_GROUP_SIZE];
        assert_eq!(unpack_3bit(&pack_3bit(&values)), values);
    }

    // -- 4-bit pack/unpack ---------------------------------------------------

    #[test]
    fn pack_unpack_4bit_identity() {
        let values: [u8; PACK_4BIT_GROUP_SIZE] = [0, MAX_4BIT_VALUE];
        let packed = pack_4bit(&values);
        let unpacked = unpack_4bit(packed);
        assert_eq!(values, unpacked);
    }

    #[test]
    fn pack_unpack_4bit_zeros() {
        let values = [0u8; PACK_4BIT_GROUP_SIZE];
        assert_eq!(unpack_4bit(pack_4bit(&values)), values);
    }

    #[test]
    fn pack_unpack_4bit_max() {
        let values = [MAX_4BIT_VALUE; PACK_4BIT_GROUP_SIZE];
        assert_eq!(unpack_4bit(pack_4bit(&values)), values);
    }

    // -- roundtrip: pack_indices_3bit / unpack_indices_3bit -------------------

    #[test]
    fn roundtrip_3bit_exact_multiple() {
        let indices: Vec<u8> = (0..TEST_3BIT_EXACT_COUNT as u8)
            .map(|i| i % (MAX_3BIT_VALUE + 1))
            .collect();
        let packed = pack_indices_3bit(&indices);
        let unpacked = unpack_indices_3bit(&packed, indices.len());
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn roundtrip_3bit_with_remainder() {
        let indices: Vec<u8> = (0..TEST_3BIT_REMAINDER_COUNT as u8)
            .map(|i| i % (MAX_3BIT_VALUE + 1))
            .collect();
        let packed = pack_indices_3bit(&indices);
        let unpacked = unpack_indices_3bit(&packed, indices.len());
        assert_eq!(indices, unpacked);
    }

    // -- roundtrip: pack_indices_4bit / unpack_indices_4bit -------------------

    #[test]
    fn roundtrip_4bit_even_count() {
        let indices: Vec<u8> = (0..TEST_4BIT_EVEN_COUNT as u8)
            .map(|i| i % TEST_4BIT_LEVELS)
            .collect();
        let packed = pack_indices_4bit(&indices);
        let unpacked = unpack_indices_4bit(&packed, indices.len());
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn roundtrip_4bit_odd_count() {
        let indices: Vec<u8> = (0..TEST_4BIT_ODD_COUNT as u8)
            .map(|i| i % TEST_4BIT_LEVELS)
            .collect();
        let packed = pack_indices_4bit(&indices);
        let unpacked = unpack_indices_4bit(&packed, indices.len());
        assert_eq!(indices, unpacked);
    }

    // -- config validation ---------------------------------------------------

    #[test]
    fn config_rejects_invalid_bits() {
        assert!(TurboQuantConfig::new(1, TEST_BLOCK_SIZE).is_err());
        assert!(TurboQuantConfig::new(5, TEST_BLOCK_SIZE).is_err());
    }

    #[test]
    fn config_rejects_non_power_of_two() {
        assert!(TurboQuantConfig::new(BITS_TQ3, 33).is_err());
        assert!(TurboQuantConfig::new(BITS_TQ4, 0).is_err());
    }

    #[test]
    fn config_accepts_valid() {
        assert!(TurboQuantConfig::new(BITS_TQ2, TEST_BLOCK_SIZE).is_ok());
        assert!(TurboQuantConfig::new(BITS_TQ3, TEST_BLOCK_SIZE).is_ok());
        assert!(TurboQuantConfig::new(BITS_TQ4, TEST_DIM_128).is_ok());
    }

    // -- size_bytes -----------------------------------------------------------

    /// Expected size for 3-bit packing of TEST_BLOCK_SIZE=32 indices:
    /// packed = 32 * 3 / 8 = 12 bytes, + 2 (scale) = 14 bytes.
    const TQ3_D32_EXPECTED_SIZE: usize = SCALE_SIZE_BYTES + 12;

    /// Expected size for 4-bit packing of TEST_BLOCK_SIZE=32 indices:
    /// packed = 32 / 2 = 16 bytes, + 2 (scale) = 18 bytes.
    const TQ4_D32_EXPECTED_SIZE: usize = SCALE_SIZE_BYTES + 16;

    #[test]
    fn packed_block_tq3_size_bytes() {
        let indices = vec![0u8; TEST_BLOCK_SIZE];
        let block = PackedBlock::new(BITS_TQ3, f16::from_f32(1.0), &indices);
        // 32 indices * 3 bits / 8 = 12 packed bytes + 2 scale bytes = 14
        assert_eq!(block.size_bytes(), TQ3_D32_EXPECTED_SIZE);
    }

    #[test]
    fn packed_block_tq4_size_bytes() {
        let indices = vec![0u8; TEST_BLOCK_SIZE];
        let block = PackedBlock::new(BITS_TQ4, f16::from_f32(1.0), &indices);
        // 32 indices / 2 = 16 packed bytes + 2 scale bytes = 18
        assert_eq!(block.size_bytes(), TQ4_D32_EXPECTED_SIZE);
    }

    // -- 2-bit pack/unpack ---------------------------------------------------

    #[test]
    fn pack_unpack_2bit_identity() {
        let values: [u8; PACK_2BIT_GROUP_SIZE] = [0, 1, 2, MAX_2BIT_VALUE];
        let packed = pack_2bit(&values);
        let unpacked = unpack_2bit(packed);
        assert_eq!(values, unpacked);
    }

    #[test]
    fn pack_unpack_2bit_zeros() {
        let values = [0u8; PACK_2BIT_GROUP_SIZE];
        assert_eq!(unpack_2bit(pack_2bit(&values)), values);
    }

    #[test]
    fn pack_unpack_2bit_max() {
        let values = [MAX_2BIT_VALUE; PACK_2BIT_GROUP_SIZE];
        assert_eq!(unpack_2bit(pack_2bit(&values)), values);
    }

    // -- roundtrip: pack_indices_2bit / unpack_indices_2bit -------------------

    #[test]
    fn roundtrip_2bit_exact_multiple() {
        let indices: Vec<u8> = (0..TEST_2BIT_EXACT_COUNT as u8)
            .map(|i| i % (MAX_2BIT_VALUE + 1))
            .collect();
        let packed = pack_indices_2bit(&indices);
        let unpacked = unpack_indices_2bit(&packed, indices.len());
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn roundtrip_2bit_with_remainder() {
        let indices: Vec<u8> = (0..TEST_2BIT_REMAINDER_COUNT as u8)
            .map(|i| i % (MAX_2BIT_VALUE + 1))
            .collect();
        let packed = pack_indices_2bit(&indices);
        let unpacked = unpack_indices_2bit(&packed, indices.len());
        assert_eq!(indices, unpacked);
    }

    // -- PackedBlock size_bytes for TQ2 --------------------------------------

    #[test]
    fn packed_block_tq2_size_bytes() {
        let indices = vec![0u8; TEST_BLOCK_SIZE];
        let block = PackedBlock::new(BITS_TQ2, f16::from_f32(1.0), &indices);
        // 32 indices / 4 per byte = 8 bytes packed + 2 bytes scale = 10
        assert_eq!(block.size_bytes(), 10);
    }
}
