//! Shared test utilities. Only compiled in test builds.

/// LCG multiplier for pseudo-random vector generation.
pub const LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;
/// LCG increment.
pub const LCG_INCREMENT: u64 = 1;
/// Right-shift for extracting bits from LCG state.
pub const LCG_SHIFT: u32 = 33;

/// Deterministic pseudo-random vector for tests. Uses LCG, no rand dependency.
pub fn pseudo_random_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..dim)
        .map(|_| {
            state = state
                .wrapping_mul(LCG_MULTIPLIER)
                .wrapping_add(LCG_INCREMENT);
            let bits = (state >> LCG_SHIFT) as i32;
            bits as f32 / (i32::MAX as f32)
        })
        .collect()
}
