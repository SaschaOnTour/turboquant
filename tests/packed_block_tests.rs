//! PackedBlock roundtrip tests: pack → unpack preserves indices and scale.
//!
//! Extracted from the former `packed_tests.rs`.

use half::f16;
use turboquant::packed::PackedBlock;

/// Representative scale factors for the roundtrip fixtures; the exact
/// magnitude is not meaningful — the assertion only verifies that the
/// stored scale survives pack/unpack.
const SCALE_TQ2: f32 = 1.23;
const SCALE_TQ3: f32 = 3.25;
const SCALE_TQ4: f32 = 2.71;
const BLOCK_LEN: usize = 64;

/// Parameterized roundtrip over all supported bit widths.
///
/// Verifies that `PackedBlock::new(bits, scale, indices)` followed by
/// `block.unpack(len)` returns the original indices and preserves the scale.
#[test]
fn packed_block_roundtrip_all_bit_widths() {
    for (bits, scale_f32, modulus) in [(2u8, SCALE_TQ2, 4u8), (3, SCALE_TQ3, 8), (4, SCALE_TQ4, 16)]
    {
        let indices: Vec<u8> = (0..BLOCK_LEN).map(|i| (i as u8) % modulus).collect();
        let scale = f16::from_f32(scale_f32);
        let block = PackedBlock::new(bits, scale, &indices);
        let recovered = block.unpack(BLOCK_LEN);
        assert_eq!(indices, recovered, "roundtrip failed for bits={bits}");
        assert_eq!(block.scale, scale, "scale drift for bits={bits}");
    }
}
