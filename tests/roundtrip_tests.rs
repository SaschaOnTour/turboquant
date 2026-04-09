//! Roundtrip tests for rotation, packed, and related modules.

mod rotation_tests {
    use approx::assert_abs_diff_eq;
    use turboquant::rotation::{generate_sign_pattern, rotate, wht_inplace, RotationOrder};

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Computes the L2 norm of a slice.
    fn l2_norm(data: &[f32]) -> f32 {
        data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Returns a deterministic pseudo-random vector of length `dim`.
    /// Uses a simple LCG so tests are reproducible without pulling in `rand`.
    fn pseudo_random_vec(dim: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..dim)
            .map(|_| {
                // LCG parameters from Numerical Recipes
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                // Map to roughly [-1, 1]
                let bits = (state >> 33) as i32;
                bits as f32 / (i32::MAX as f32)
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // WHT norm preservation
    // -----------------------------------------------------------------------

    #[test]
    fn wht_preserves_norm_dim64() {
        wht_preserves_norm(64);
    }

    #[test]
    fn wht_preserves_norm_dim128() {
        wht_preserves_norm(128);
    }

    #[test]
    fn wht_preserves_norm_dim256() {
        wht_preserves_norm(256);
    }

    fn wht_preserves_norm(dim: usize) {
        let mut data = pseudo_random_vec(dim, 12345);
        let norm_before = l2_norm(&data);

        wht_inplace(&mut data);
        let norm_after = l2_norm(&data);

        assert_abs_diff_eq!(norm_before, norm_after, epsilon = 1e-3);
    }

    // -----------------------------------------------------------------------
    // WHT self-inversity
    // -----------------------------------------------------------------------

    #[test]
    fn wht_is_self_inverse_dim64() {
        wht_is_self_inverse(64);
    }

    #[test]
    fn wht_is_self_inverse_dim128() {
        wht_is_self_inverse(128);
    }

    #[test]
    fn wht_is_self_inverse_dim256() {
        wht_is_self_inverse(256);
    }

    fn wht_is_self_inverse(dim: usize) {
        let original = pseudo_random_vec(dim, 54321);
        let mut data = original.clone();

        wht_inplace(&mut data);
        wht_inplace(&mut data);

        for (a, b) in original.iter().zip(data.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-4);
        }
    }

    // -----------------------------------------------------------------------
    // validate_rotation_inputs rejects non-power-of-two (via rotate)
    // -----------------------------------------------------------------------

    #[test]
    fn validate_rotation_rejects_non_power_of_two() {
        let mut data = vec![1.0; 3];
        let signs = vec![1.0; 3];
        assert!(rotate(&mut data, &signs, RotationOrder::Forward).is_err());
    }

    #[test]
    fn validate_rotation_accepts_power_of_two() {
        let mut data = vec![1.0; 8];
        let signs = generate_sign_pattern(8, 42);
        assert!(rotate(&mut data, &signs, RotationOrder::Forward).is_ok());
    }

    // -----------------------------------------------------------------------
    // Sign-pattern determinism
    // -----------------------------------------------------------------------

    #[test]
    fn same_seed_produces_same_sign_pattern() {
        let a = generate_sign_pattern(256, 42);
        let b = generate_sign_pattern(256, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn different_seeds_produce_different_sign_patterns() {
        let a = generate_sign_pattern(256, 1);
        let b = generate_sign_pattern(256, 2);
        // They could theoretically match, but with 256 elements it is
        // astronomically unlikely.
        assert_ne!(a, b);
    }

    #[test]
    fn sign_pattern_contains_only_plus_minus_one() {
        let pattern = generate_sign_pattern(512, 77);
        for &v in &pattern {
            assert!(v == 1.0 || v == -1.0, "unexpected value: {v}");
        }
    }

    // -----------------------------------------------------------------------
    // Full rotation roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn rotation_roundtrip_dim64() {
        rotation_roundtrip(64, 100);
    }

    #[test]
    fn rotation_roundtrip_dim128() {
        rotation_roundtrip(128, 200);
    }

    #[test]
    fn rotation_roundtrip_dim256() {
        rotation_roundtrip(256, 300);
    }

    fn rotation_roundtrip(dim: usize, seed: u64) {
        let original = pseudo_random_vec(dim, seed);
        let sign_pattern = generate_sign_pattern(dim, seed);

        let mut data = original.clone();
        rotate(&mut data, &sign_pattern, RotationOrder::Forward).expect("rotate should succeed");
        rotate(&mut data, &sign_pattern, RotationOrder::Inverse)
            .expect("inverse_rotate should succeed");

        for (a, b) in original.iter().zip(data.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-4);
        }
    }

    // -----------------------------------------------------------------------
    // Rotation preserves norm
    // -----------------------------------------------------------------------

    #[test]
    fn rotation_preserves_norm() {
        let dim = 128;
        let seed = 55;
        let sign_pattern = generate_sign_pattern(dim, seed);
        let mut data = pseudo_random_vec(dim, seed);
        let norm_before = l2_norm(&data);

        rotate(&mut data, &sign_pattern, RotationOrder::Forward).expect("rotate should succeed");
        let norm_after = l2_norm(&data);

        assert_abs_diff_eq!(norm_before, norm_after, epsilon = 1e-3);
    }

    // -----------------------------------------------------------------------
    // Distribution test: rotated coordinates should have mean ~ 0
    // -----------------------------------------------------------------------

    #[test]
    fn rotated_coordinates_have_zero_mean() {
        let dim = 256;
        let num_samples = 50;
        let mut total_mean = 0.0_f64;

        for sample_seed in 0..num_samples {
            let sign_pattern = generate_sign_pattern(dim, 999);
            let mut data = pseudo_random_vec(dim, 1000 + sample_seed);

            // Normalize to unit vector
            let norm = l2_norm(&data);
            if norm > 0.0 {
                for v in data.iter_mut() {
                    *v /= norm;
                }
            }

            rotate(&mut data, &sign_pattern, RotationOrder::Forward)
                .expect("rotate should succeed");

            let mean: f64 = data.iter().map(|&x| x as f64).sum::<f64>() / dim as f64;
            total_mean += mean;
        }

        let avg_mean = total_mean / num_samples as f64;
        assert!(
            avg_mean.abs() < 0.05,
            "average mean across samples should be near zero, got {avg_mean}"
        );
    }

    // -----------------------------------------------------------------------
    // Distribution test: variance of rotated unit vectors
    // -----------------------------------------------------------------------

    #[test]
    fn rotated_unit_vector_has_expected_variance() {
        let dim = 256;
        let sign_pattern = generate_sign_pattern(dim, 7777);
        let mut data = pseudo_random_vec(dim, 8888);

        // Normalize to unit vector
        let norm = l2_norm(&data);
        for v in data.iter_mut() {
            *v /= norm;
        }

        rotate(&mut data, &sign_pattern, RotationOrder::Forward).expect("rotate should succeed");

        // For a rotated unit vector, each coordinate has variance 1/d
        let expected_variance = 1.0_f64 / dim as f64;
        let mean: f64 = data.iter().map(|&x| x as f64).sum::<f64>() / dim as f64;
        let variance: f64 = data
            .iter()
            .map(|&x| {
                let diff = x as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / dim as f64;

        // The variance should be close to 1/d = 0.00390625 for d=256.
        // Allow generous tolerance since this is a single sample.
        assert_abs_diff_eq!(variance, expected_variance, epsilon = 0.005);
    }

    // -----------------------------------------------------------------------
    // Error cases
    // -----------------------------------------------------------------------

    #[test]
    fn rotate_rejects_non_power_of_two() {
        let mut data = vec![1.0; 5];
        let signs = vec![1.0; 5];
        assert!(rotate(&mut data, &signs, RotationOrder::Forward).is_err());
    }

    #[test]
    fn rotate_rejects_dimension_mismatch() {
        let mut data = vec![1.0; 8];
        let signs = vec![1.0; 4];
        assert!(rotate(&mut data, &signs, RotationOrder::Forward).is_err());
    }

    #[test]
    fn inverse_rotate_rejects_non_power_of_two() {
        let mut data = vec![1.0; 6];
        let signs = vec![1.0; 6];
        assert!(rotate(&mut data, &signs, RotationOrder::Inverse).is_err());
    }

    #[test]
    fn inverse_rotate_rejects_dimension_mismatch() {
        let mut data = vec![1.0; 16];
        let signs = vec![1.0; 8];
        assert!(rotate(&mut data, &signs, RotationOrder::Inverse).is_err());
    }
}

mod packed_tests {
    use half::f16;
    use turboquant::packed::{
        pack_2bit, pack_3bit, pack_4bit, pack_indices_2bit, pack_indices_3bit, pack_indices_4bit,
        unpack_2bit, unpack_3bit, unpack_4bit, unpack_indices_2bit, unpack_indices_3bit,
        unpack_indices_4bit, PackedBlock, TurboQuantConfig,
    };

    // ----- 3-bit roundtrip ---------------------------------------------------

    #[test]
    fn roundtrip_3bit_all_valid_values() {
        // Every combination of 0..=7 in the first two slots, fixed elsewhere.
        for a in 0u8..=7 {
            for b in 0u8..=7 {
                let values: [u8; 8] = [a, b, 0, 7, 3, 5, 1, 6];
                let packed = pack_3bit(&values);
                let unpacked = unpack_3bit(&packed);
                assert_eq!(values, unpacked, "failed for a={a}, b={b}");
            }
        }
    }

    #[test]
    fn roundtrip_3bit_all_zeros() {
        let values = [0u8; 8];
        assert_eq!(unpack_3bit(&pack_3bit(&values)), values);
    }

    #[test]
    fn roundtrip_3bit_all_max() {
        let values = [7u8; 8];
        assert_eq!(unpack_3bit(&pack_3bit(&values)), values);
    }

    #[test]
    fn roundtrip_3bit_mixed() {
        let values: [u8; 8] = [1, 3, 5, 7, 0, 2, 4, 6];
        assert_eq!(unpack_3bit(&pack_3bit(&values)), values);
    }

    // ----- 4-bit roundtrip ---------------------------------------------------

    #[test]
    fn roundtrip_4bit_all_valid_values() {
        for a in 0u8..=15 {
            for b in 0u8..=15 {
                let values: [u8; 2] = [a, b];
                let packed = pack_4bit(&values);
                let unpacked = unpack_4bit(packed);
                assert_eq!(values, unpacked, "failed for a={a}, b={b}");
            }
        }
    }

    #[test]
    fn roundtrip_4bit_all_zeros() {
        let values = [0u8; 2];
        assert_eq!(unpack_4bit(pack_4bit(&values)), values);
    }

    #[test]
    fn roundtrip_4bit_all_max() {
        let values = [15u8; 2];
        assert_eq!(unpack_4bit(pack_4bit(&values)), values);
    }

    #[test]
    fn roundtrip_4bit_mixed() {
        let values: [u8; 2] = [3, 12];
        assert_eq!(unpack_4bit(pack_4bit(&values)), values);
    }

    // ----- size_bytes --------------------------------------------------------

    #[test]
    fn packed_block_tq3_size_bytes_dim_32() {
        // 32 indices / 8 per group = 4 groups * 3 bytes = 12 bytes packed
        // total = 2 (scale) + 12 = 14
        let indices = vec![0u8; 32];
        let block = PackedBlock::new(3, f16::from_f32(1.0), &indices);
        assert_eq!(block.size_bytes(), 14);
    }

    #[test]
    fn packed_block_tq3_size_bytes_dim_128() {
        // 128 / 8 = 16 groups * 3 = 48 bytes packed => total 50
        let indices = vec![3u8; 128];
        let block = PackedBlock::new(3, f16::from_f32(2.5), &indices);
        assert_eq!(block.size_bytes(), 50);
    }

    #[test]
    fn packed_block_tq4_size_bytes_dim_32() {
        // 32 indices / 2 = 16 bytes packed => total 18
        let indices = vec![0u8; 32];
        let block = PackedBlock::new(4, f16::from_f32(1.0), &indices);
        assert_eq!(block.size_bytes(), 18);
    }

    #[test]
    fn packed_block_tq4_size_bytes_dim_128() {
        // 128 / 2 = 64 bytes packed => total 66
        let indices = vec![9u8; 128];
        let block = PackedBlock::new(4, f16::from_f32(0.5), &indices);
        assert_eq!(block.size_bytes(), 66);
    }

    // ----- TurboQuantConfig validation --------------------------------------

    #[test]
    fn config_accepts_bits_2() {
        assert!(TurboQuantConfig::new(2, 64).is_ok());
    }

    #[test]
    fn config_rejects_bits_1() {
        assert!(TurboQuantConfig::new(1, 64).is_err());
    }

    #[test]
    fn config_rejects_bits_5() {
        assert!(TurboQuantConfig::new(5, 64).is_err());
    }

    #[test]
    fn config_rejects_non_power_of_two() {
        assert!(TurboQuantConfig::new(3, 33).is_err());
        assert!(TurboQuantConfig::new(4, 100).is_err());
    }

    #[test]
    fn config_rejects_dim_zero() {
        assert!(TurboQuantConfig::new(3, 0).is_err());
    }

    #[test]
    fn config_accepts_valid_3bit() {
        // Validates that new(3, 64) succeeds -- the config is usable for quantization.
        let _cfg = TurboQuantConfig::new(3, 64).unwrap();
    }

    #[test]
    fn config_accepts_valid_4bit() {
        // Validates that new(4, 256) succeeds -- the config is usable for quantization.
        let _cfg = TurboQuantConfig::new(4, 256).unwrap();
    }

    // ----- Full vector roundtrip (128 elements) ------------------------------

    #[test]
    fn full_vector_roundtrip_3bit_128() {
        let indices: Vec<u8> = (0..128).map(|i| (i % 8) as u8).collect();
        let packed = pack_indices_3bit(&indices);
        let unpacked = unpack_indices_3bit(&packed, 128);
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn full_vector_roundtrip_4bit_128() {
        let indices: Vec<u8> = (0..128).map(|i| (i % 16) as u8).collect();
        let packed = pack_indices_4bit(&indices);
        let unpacked = unpack_indices_4bit(&packed, 128);
        assert_eq!(indices, unpacked);
    }

    // ----- Block roundtrip ---------------------------------------------------

    #[test]
    fn packed_block_tq3_roundtrip() {
        let indices: Vec<u8> = (0..64).map(|i| (i % 8) as u8).collect();
        let scale = f16::from_f32(3.25);
        let block = PackedBlock::new(3, scale, &indices);
        let recovered = block.unpack(64);
        assert_eq!(indices, recovered);
        assert_eq!(block.scale, scale);
    }

    #[test]
    fn packed_block_tq4_roundtrip() {
        let indices: Vec<u8> = (0..64).map(|i| (i % 16) as u8).collect();
        let scale = f16::from_f32(2.71);
        let block = PackedBlock::new(4, scale, &indices);
        let recovered = block.unpack(64);
        assert_eq!(indices, recovered);
        assert_eq!(block.scale, scale);
    }

    // ----- 2-bit roundtrip ---------------------------------------------------

    #[test]
    fn roundtrip_2bit_all_valid_values() {
        for a in 0u8..=3 {
            for b in 0u8..=3 {
                for c in 0u8..=3 {
                    for d in 0u8..=3 {
                        let values: [u8; 4] = [a, b, c, d];
                        let packed = pack_2bit(&values);
                        let unpacked = unpack_2bit(packed);
                        assert_eq!(values, unpacked, "failed for a={a}, b={b}, c={c}, d={d}");
                    }
                }
            }
        }
    }

    #[test]
    fn roundtrip_2bit_all_zeros() {
        let values = [0u8; 4];
        assert_eq!(unpack_2bit(pack_2bit(&values)), values);
    }

    #[test]
    fn roundtrip_2bit_all_max() {
        let values = [3u8; 4];
        assert_eq!(unpack_2bit(pack_2bit(&values)), values);
    }

    #[test]
    fn full_vector_roundtrip_2bit_128() {
        let indices: Vec<u8> = (0..128).map(|i| (i % 4) as u8).collect();
        let packed = pack_indices_2bit(&indices);
        let unpacked = unpack_indices_2bit(&packed, 128);
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn packed_block_tq2_roundtrip() {
        let indices: Vec<u8> = (0..64).map(|i| (i % 4) as u8).collect();
        let scale = f16::from_f32(1.23);
        let block = PackedBlock::new(2, scale, &indices);
        let recovered = block.unpack(64);
        assert_eq!(indices, recovered);
        assert_eq!(block.scale, scale);
    }

    #[test]
    fn packed_block_tq2_size_bytes_dim_32() {
        // 32 indices / 4 per byte = 8 bytes packed
        // total = 2 (scale) + 8 = 10
        let indices = vec![0u8; 32];
        let block = PackedBlock::new(2, f16::from_f32(1.0), &indices);
        assert_eq!(block.size_bytes(), 10);
    }

    #[test]
    fn packed_block_tq2_size_bytes_dim_128() {
        // 128 / 4 = 32 bytes packed => total 34
        let indices = vec![1u8; 128];
        let block = PackedBlock::new(2, f16::from_f32(2.5), &indices);
        assert_eq!(block.size_bytes(), 34);
    }

    #[test]
    fn config_accepts_valid_2bit() {
        // Validates that new(2, 64) succeeds -- the config is usable for quantization.
        let _cfg = TurboQuantConfig::new(2, 64).unwrap();
    }
}

mod quantize_tests {
    use approx::assert_abs_diff_eq;
    use turboquant::packed::TurboQuantConfig;
    use turboquant::quantize::{dequantize_rotated, dequantize_vec, l2_norm, quantize_vec};

    // -----------------------------------------------------------------------
    // Constants
    // -----------------------------------------------------------------------

    /// Default seed for rotation.
    const TEST_SEED: u64 = 42;
    /// Tolerance for norm comparisons after quantization roundtrip.
    /// 3-bit quantization introduces ~18% relative error on average (sqrt(MSE=0.034)),
    /// so the norm can deviate significantly.  f16 rounding adds further noise.
    const NORM_EPSILON: f32 = 0.35;
    /// Tolerance for near-zero checks.
    const ZERO_EPSILON: f32 = 0.1;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Returns a deterministic pseudo-random vector of length `dim`.
    fn pseudo_random_vec(dim: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..dim)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let bits = (state >> 33) as i32;
                bits as f32 / (i32::MAX as f32)
            })
            .collect()
    }

    /// Computes the squared error between two vectors.
    fn squared_error(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y) * (x - y))
            .sum()
    }

    // -----------------------------------------------------------------------
    // Roundtrip: dequantize(quantize(x)) is close to x
    // -----------------------------------------------------------------------

    #[test]
    fn roundtrip_tq3_dim64() {
        roundtrip_check(3, 64, 1000);
    }

    #[test]
    fn roundtrip_tq3_dim128() {
        roundtrip_check(3, 128, 2000);
    }

    #[test]
    fn roundtrip_tq3_dim256() {
        roundtrip_check(3, 256, 3000);
    }

    #[test]
    fn roundtrip_tq4_dim64() {
        roundtrip_check(4, 64, 4000);
    }

    #[test]
    fn roundtrip_tq4_dim128() {
        roundtrip_check(4, 128, 5000);
    }

    #[test]
    fn roundtrip_tq4_dim256() {
        roundtrip_check(4, 256, 6000);
    }

    fn roundtrip_check(bits: u8, dim: usize, seed: u64) {
        let config = TurboQuantConfig::new(bits, dim)
            .unwrap()
            .with_seed(TEST_SEED);
        let data = pseudo_random_vec(dim, seed);
        let block = quantize_vec(&config, &data).unwrap();
        let recovered = dequantize_vec(&config, &block).unwrap();

        let orig_norm_sq = data.iter().map(|&x| x * x).sum::<f32>();
        let err_sq = squared_error(&data, &recovered);
        let relative_mse = err_sq / orig_norm_sq;

        // Single-vector relative MSE can be much higher than the aggregate
        // mean (0.034 for TQ3, 0.009 for TQ4, ~0.10 for TQ2). The proper
        // quality gate is mse_validation.rs which checks over 10,000 vectors.
        let threshold = match bits {
            2 => 1.5,
            3 => 1.0,
            _ => 0.5,
        };
        assert!(
            relative_mse < threshold,
            "bits={bits}, dim={dim}: relative MSE {relative_mse} exceeds {threshold}"
        );
    }

    // -----------------------------------------------------------------------
    // Null vector: quantize([0,...,0]) doesn't panic, dequantize gives zeros
    // -----------------------------------------------------------------------

    #[test]
    fn null_vector_tq3() {
        null_vector_check(3, 128);
    }

    #[test]
    fn null_vector_tq4() {
        null_vector_check(4, 128);
    }

    fn null_vector_check(bits: u8, dim: usize) {
        let config = TurboQuantConfig::new(bits, dim)
            .unwrap()
            .with_seed(TEST_SEED);
        let data = vec![0.0_f32; dim];
        let block = quantize_vec(&config, &data).unwrap();
        let recovered = dequantize_vec(&config, &block).unwrap();
        let norm = l2_norm(&recovered);
        assert!(
            norm < ZERO_EPSILON,
            "null vector roundtrip should give near-zero, got norm={norm}"
        );
    }

    // -----------------------------------------------------------------------
    // Unit vector: quantize(e1) works correctly
    // -----------------------------------------------------------------------

    #[test]
    fn unit_vector_tq3() {
        unit_vector_check(3, 128);
    }

    #[test]
    fn unit_vector_tq4() {
        unit_vector_check(4, 128);
    }

    fn unit_vector_check(bits: u8, dim: usize) {
        let config = TurboQuantConfig::new(bits, dim)
            .unwrap()
            .with_seed(TEST_SEED);
        let mut data = vec![0.0_f32; dim];
        data[0] = 1.0;
        let block = quantize_vec(&config, &data).unwrap();
        let recovered = dequantize_vec(&config, &block).unwrap();

        // The recovered vector should have a non-zero norm in the right
        // ballpark.  Exact norm preservation is not guaranteed by scalar
        // quantization.
        let rec_norm = l2_norm(&recovered);
        assert!(rec_norm > 0.3, "recovered norm too small: {rec_norm}");
        assert!(rec_norm < 2.0, "recovered norm too large: {rec_norm}");
    }

    // -----------------------------------------------------------------------
    // Constant vector: all same value
    // -----------------------------------------------------------------------

    #[test]
    fn constant_vector_tq3() {
        constant_vector_check(3, 128);
    }

    #[test]
    fn constant_vector_tq4() {
        constant_vector_check(4, 128);
    }

    fn constant_vector_check(bits: u8, dim: usize) {
        let config = TurboQuantConfig::new(bits, dim)
            .unwrap()
            .with_seed(TEST_SEED);
        let val = 0.5_f32;
        let data = vec![val; dim];
        let block = quantize_vec(&config, &data).unwrap();
        let recovered = dequantize_vec(&config, &block).unwrap();

        // Verify the pipeline doesn't blow up on constant vectors and the
        // recovered norm is in a reasonable range.
        let orig_norm = l2_norm(&data);
        let rec_norm = l2_norm(&recovered);
        let ratio = rec_norm / orig_norm;
        // Constant vectors are adversarial for rotation-based quantization
        // (all energy concentrates in one WHT coefficient), so the ratio
        // can be quite low.
        assert!(ratio > 0.1, "recovered norm too small: ratio={ratio}");
        assert!(ratio < 3.0, "recovered norm too large: ratio={ratio}");
    }

    // -----------------------------------------------------------------------
    // Determinism: same input + config -> identical output
    // -----------------------------------------------------------------------

    #[test]
    fn determinism_tq3() {
        determinism_check(3, 128, 11111);
    }

    #[test]
    fn determinism_tq4() {
        determinism_check(4, 128, 22222);
    }

    fn determinism_check(bits: u8, dim: usize, seed: u64) {
        let config = TurboQuantConfig::new(bits, dim)
            .unwrap()
            .with_seed(TEST_SEED);
        let data = pseudo_random_vec(dim, seed);

        let block_a = quantize_vec(&config, &data).unwrap();
        let block_b = quantize_vec(&config, &data).unwrap();

        let rec_a = dequantize_vec(&config, &block_a).unwrap();
        let rec_b = dequantize_vec(&config, &block_b).unwrap();

        assert_eq!(rec_a, rec_b, "quantization should be deterministic");
    }

    // -----------------------------------------------------------------------
    // Different dimensions: d=64, d=128, d=256
    // -----------------------------------------------------------------------

    #[test]
    fn different_dimensions_tq3() {
        for &dim in &[64, 128, 256] {
            let config = TurboQuantConfig::new(3, dim).unwrap().with_seed(TEST_SEED);
            let data = pseudo_random_vec(dim, dim as u64);
            let block = quantize_vec(&config, &data).unwrap();
            let recovered = dequantize_vec(&config, &block).unwrap();
            assert_eq!(recovered.len(), dim);
        }
    }

    #[test]
    fn different_dimensions_tq4() {
        for &dim in &[64, 128, 256] {
            let config = TurboQuantConfig::new(4, dim).unwrap().with_seed(TEST_SEED);
            let data = pseudo_random_vec(dim, dim as u64 + 1000);
            let block = quantize_vec(&config, &data).unwrap();
            let recovered = dequantize_vec(&config, &block).unwrap();
            assert_eq!(recovered.len(), dim);
        }
    }

    // -----------------------------------------------------------------------
    // dequantize_rotated: differs from full dequantize but same norm
    // -----------------------------------------------------------------------

    #[test]
    fn dequantize_rotated_differs_but_same_norm_tq3() {
        dequantize_rotated_check(3, 128, 33333);
    }

    #[test]
    fn dequantize_rotated_differs_but_same_norm_tq4() {
        dequantize_rotated_check(4, 128, 44444);
    }

    fn dequantize_rotated_check(bits: u8, dim: usize, seed: u64) {
        let config = TurboQuantConfig::new(bits, dim)
            .unwrap()
            .with_seed(TEST_SEED);
        let data = pseudo_random_vec(dim, seed);
        let block = quantize_vec(&config, &data).unwrap();

        let full = dequantize_vec(&config, &block).unwrap();
        let rotated = dequantize_rotated(&config, &block).unwrap();

        // Coordinates should differ.
        assert_ne!(full, rotated, "rotated and full dequantize should differ");

        // Norms should be approximately equal (rotation preserves norm).
        let full_norm = l2_norm(&full);
        let rotated_norm = l2_norm(&rotated);
        assert_abs_diff_eq!(full_norm, rotated_norm, epsilon = NORM_EPSILON);
    }

    // -----------------------------------------------------------------------
    // PackedBlock: both TQ2, TQ3, and TQ4 work via quantize_vec
    // -----------------------------------------------------------------------

    #[test]
    fn packed_block_tq2() {
        let config = TurboQuantConfig::new(2, 64).unwrap().with_seed(TEST_SEED);
        let data = pseudo_random_vec(64, 44444);
        let block = quantize_vec(&config, &data).unwrap();
        assert_eq!(block.bits, 2);
        let recovered = dequantize_vec(&config, &block).unwrap();
        assert_eq!(recovered.len(), 64);
    }

    #[test]
    fn packed_block_tq3() {
        let config = TurboQuantConfig::new(3, 64).unwrap().with_seed(TEST_SEED);
        let data = pseudo_random_vec(64, 55555);
        let block = quantize_vec(&config, &data).unwrap();
        assert_eq!(block.bits, 3);
        let recovered = dequantize_vec(&config, &block).unwrap();
        assert_eq!(recovered.len(), 64);
    }

    #[test]
    fn packed_block_tq4() {
        let config = TurboQuantConfig::new(4, 64).unwrap().with_seed(TEST_SEED);
        let data = pseudo_random_vec(64, 66666);
        let block = quantize_vec(&config, &data).unwrap();
        assert_eq!(block.bits, 4);
        let recovered = dequantize_vec(&config, &block).unwrap();
        assert_eq!(recovered.len(), 64);
    }

    // -----------------------------------------------------------------------
    // 2-bit roundtrip tests
    // -----------------------------------------------------------------------

    #[test]
    fn roundtrip_tq2_dim64() {
        roundtrip_check(2, 64, 7000);
    }

    #[test]
    fn roundtrip_tq2_dim128() {
        roundtrip_check(2, 128, 8000);
    }

    #[test]
    fn roundtrip_tq2_dim256() {
        roundtrip_check(2, 256, 9000);
    }

    #[test]
    fn null_vector_tq2() {
        null_vector_check(2, 128);
    }

    #[test]
    fn unit_vector_tq2() {
        unit_vector_check(2, 128);
    }

    #[test]
    fn constant_vector_tq2() {
        constant_vector_check(2, 128);
    }

    #[test]
    fn determinism_tq2() {
        determinism_check(2, 128, 33333);
    }

    #[test]
    fn different_dimensions_tq2() {
        for &dim in &[64, 128, 256] {
            let config = TurboQuantConfig::new(2, dim).unwrap().with_seed(TEST_SEED);
            let data = pseudo_random_vec(dim, dim as u64 + 2000);
            let block = quantize_vec(&config, &data).unwrap();
            let recovered = dequantize_vec(&config, &block).unwrap();
            assert_eq!(recovered.len(), dim);
        }
    }

    #[test]
    fn dequantize_rotated_differs_but_same_norm_tq2() {
        dequantize_rotated_check(2, 128, 55555);
    }
}
