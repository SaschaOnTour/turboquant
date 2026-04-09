//! Vector-level packing and unpacking of quantized indices.
//!
//! These functions pack/unpack entire vectors of indices using the element-level
//! primitives from the parent module.

use super::{
    pack_2bit, pack_3bit, pack_4bit, unpack_2bit, unpack_3bit, unpack_4bit, PACK_2BIT_GROUP_SIZE,
    PACK_3BIT_BYTES, PACK_3BIT_GROUP_SIZE, PACK_4BIT_GROUP_SIZE,
};

// ---------------------------------------------------------------------------
// 2-bit vector-level helpers  (pure Operation -- arithmetic/logic only)
// ---------------------------------------------------------------------------

/// Compute the number of full groups of 4 that fit in `len` elements.
///
/// Pure Operation: arithmetic only.
pub(super) fn num_2bit_groups(len: usize) -> usize {
    len / PACK_2BIT_GROUP_SIZE
}

/// Check whether `len` elements have a remainder after grouping by 4.
///
/// Pure Operation: arithmetic only.
pub(super) fn has_2bit_remainder(len: usize) -> bool {
    len % PACK_2BIT_GROUP_SIZE != 0
}

/// Compute capacity for the packed 2-bit byte vector.
///
/// Pure Operation: arithmetic only.
pub(super) fn packed_2bit_capacity(num_groups: usize, has_remainder: bool) -> usize {
    num_groups + usize::from(has_remainder)
}

/// Convert a chunk of exactly 4 bytes into the fixed-size array expected by
/// `pack_2bit`.
///
/// Pure Operation: slice-to-array conversion only.
pub(super) fn chunk_to_2bit_array(chunk: &[u8]) -> [u8; PACK_2BIT_GROUP_SIZE] {
    chunk.try_into().expect("chunk size matches group size")
}

/// Pad a remainder slice (< 4 elements) into a full 4-element array, filling
/// the tail with zeros.
///
/// Pure Operation: copy only.
pub(super) fn pad_remainder_2bit(tail: &[u8]) -> [u8; PACK_2BIT_GROUP_SIZE] {
    let mut padded = [0u8; PACK_2BIT_GROUP_SIZE];
    padded[..tail.len()].copy_from_slice(tail);
    padded
}

// ---------------------------------------------------------------------------
// 3-bit vector-level helpers  (pure Operation -- arithmetic/logic only)
// ---------------------------------------------------------------------------

/// Compute the number of full groups of 8 that fit in `len` elements.
///
/// Pure Operation: arithmetic only.
pub(super) fn num_3bit_groups(len: usize) -> usize {
    len / PACK_3BIT_GROUP_SIZE
}

/// Check whether `len` elements have a remainder after grouping by 8.
///
/// Pure Operation: arithmetic only.
pub(super) fn has_3bit_remainder(len: usize) -> bool {
    len % PACK_3BIT_GROUP_SIZE != 0
}

/// Compute capacity for the packed 3-bit byte vector.
///
/// Pure Operation: arithmetic only, no calls to other project functions.
pub(super) fn packed_3bit_capacity(num_groups: usize, has_remainder: bool) -> usize {
    let remainder_bytes = if has_remainder { PACK_3BIT_BYTES } else { 0 };
    num_groups * PACK_3BIT_BYTES + remainder_bytes
}

/// Convert a chunk of exactly 8 bytes into the fixed-size array expected by
/// `pack_3bit`.
///
/// Pure Operation: slice-to-array conversion only.
pub(super) fn chunk_to_3bit_array(chunk: &[u8]) -> [u8; PACK_3BIT_GROUP_SIZE] {
    chunk.try_into().expect("chunk size matches group size")
}

/// Pad a remainder slice (< 8 elements) into a full 8-element array, filling
/// the tail with zeros.
///
/// Pure Operation: copy only.
pub(super) fn pad_remainder_3bit(tail: &[u8]) -> [u8; PACK_3BIT_GROUP_SIZE] {
    let mut padded = [0u8; PACK_3BIT_GROUP_SIZE];
    padded[..tail.len()].copy_from_slice(tail);
    padded
}

/// Convert a 3-byte chunk into the fixed-size array expected by `unpack_3bit`.
///
/// Pure Operation: slice-to-array conversion only.
pub(super) fn chunk_to_packed_3bit_array(chunk: &[u8]) -> [u8; PACK_3BIT_BYTES] {
    chunk.try_into().expect("chunk size matches group size")
}

// ---------------------------------------------------------------------------
// 4-bit vector-level helpers  (pure Operation -- arithmetic/logic only)
// ---------------------------------------------------------------------------

/// Compute the number of full pairs that fit in `len` elements.
///
/// Pure Operation: arithmetic only.
pub(super) fn num_4bit_pairs(len: usize) -> usize {
    len / PACK_4BIT_GROUP_SIZE
}

/// Check whether `len` elements have a trailing odd element.
///
/// Pure Operation: arithmetic only.
pub(super) fn has_4bit_remainder(len: usize) -> bool {
    len % PACK_4BIT_GROUP_SIZE != 0
}

/// Compute capacity for the packed 4-bit byte vector.
///
/// Pure Operation: arithmetic only.
pub(super) fn packed_4bit_capacity(num_pairs: usize, has_remainder: bool) -> usize {
    num_pairs + usize::from(has_remainder)
}

/// Convert a 2-byte chunk into the fixed-size array expected by `pack_4bit`.
///
/// Pure Operation: slice-to-array conversion only.
pub(super) fn chunk_to_4bit_array(pair: &[u8]) -> [u8; PACK_4BIT_GROUP_SIZE] {
    pair.try_into().expect("chunk size matches group size")
}

/// Build the pair for packing a trailing odd element (high nibble is zero).
///
/// Pure Operation: value construction only.
pub(super) fn trailing_4bit_pair(last: u8) -> [u8; PACK_4BIT_GROUP_SIZE] {
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
