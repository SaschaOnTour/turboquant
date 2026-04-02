use approx::assert_relative_eq;
use turboquant::codebook::{beta_pdf, generate_codebook, get_codebook, nearest_centroid, Codebook};

// ---------------------------------------------------------------------------
// Helper: assert that a codebook satisfies basic structural invariants
// ---------------------------------------------------------------------------

fn assert_codebook_valid(cb: &Codebook, expected_k: usize) {
    // Correct number of centroids and boundaries.
    assert_eq!(cb.centroids.len(), expected_k);
    assert_eq!(cb.boundaries.len(), expected_k - 1);

    // All centroids in [-1, 1].
    for &c in &cb.centroids {
        assert!((-1.0..=1.0).contains(&c), "centroid {c} outside [-1, 1]");
    }

    // Centroids are sorted in strictly increasing order.
    for w in cb.centroids.windows(2) {
        assert!(w[0] < w[1], "centroids not sorted: {} >= {}", w[0], w[1]);
    }

    // Boundaries are sorted and lie between their neighbouring centroids.
    for (i, &b) in cb.boundaries.iter().enumerate() {
        assert!(
            cb.centroids[i] < b,
            "boundary {b} not above centroid {i} ({})",
            cb.centroids[i]
        );
        assert!(
            b < cb.centroids[i + 1],
            "boundary {b} not below centroid {} ({})",
            i + 1,
            cb.centroids[i + 1]
        );
    }
}

// ---------------------------------------------------------------------------
// Symmetry: centroids must be symmetric around 0
// ---------------------------------------------------------------------------

fn assert_symmetric(cb: &Codebook) {
    let k = cb.centroids.len();
    for i in 0..k / 2 {
        let j = k - 1 - i;
        assert_relative_eq!(cb.centroids[i], -cb.centroids[j], epsilon = 1e-8);
    }
    let m = cb.boundaries.len();
    for i in 0..m / 2 {
        let j = m - 1 - i;
        assert_relative_eq!(cb.boundaries[i], -cb.boundaries[j], epsilon = 1e-8);
    }
    // Middle boundary should be 0 for even k.
    if m % 2 == 1 {
        assert_relative_eq!(cb.boundaries[m / 2], 0.0, epsilon = 1e-10);
    }
}

#[test]
fn symmetry_3bit_d64() {
    let cb = get_codebook(3, 64).unwrap();
    assert_symmetric(&cb);
}

#[test]
fn symmetry_3bit_d128() {
    let cb = get_codebook(3, 128).unwrap();
    assert_symmetric(&cb);
}

#[test]
fn symmetry_3bit_d256() {
    let cb = get_codebook(3, 256).unwrap();
    assert_symmetric(&cb);
}

#[test]
fn symmetry_4bit_d64() {
    let cb = get_codebook(4, 64).unwrap();
    assert_symmetric(&cb);
}

#[test]
fn symmetry_4bit_d128() {
    let cb = get_codebook(4, 128).unwrap();
    assert_symmetric(&cb);
}

#[test]
fn symmetry_4bit_d256() {
    let cb = get_codebook(4, 256).unwrap();
    assert_symmetric(&cb);
}

// ---------------------------------------------------------------------------
// Structural validity of pre-computed codebooks
// ---------------------------------------------------------------------------

#[test]
fn valid_3bit_codebooks() {
    for dim in [64, 128, 256] {
        let cb = get_codebook(3, dim).unwrap();
        assert_codebook_valid(&cb, 8);
    }
}

#[test]
fn valid_4bit_codebooks() {
    for dim in [64, 128, 256] {
        let cb = get_codebook(4, dim).unwrap();
        assert_codebook_valid(&cb, 16);
    }
}

// ---------------------------------------------------------------------------
// generate_codebook produces a valid, symmetric result
// ---------------------------------------------------------------------------

#[test]
fn generate_3bit_d64_valid_and_symmetric() {
    let cb = generate_codebook(3, 64);
    assert_codebook_valid(&cb, 8);
    assert_symmetric(&cb);
}

#[test]
fn generate_4bit_d128_valid_and_symmetric() {
    let cb = generate_codebook(4, 128);
    assert_codebook_valid(&cb, 16);
    assert_symmetric(&cb);
}

// ---------------------------------------------------------------------------
// Monotonically decreasing MSE distortion over iterations
// ---------------------------------------------------------------------------

#[test]
fn distortion_decreases_over_iterations() {
    // We run Lloyd-Max manually for a few steps and check distortion goes down.
    // We use generate_codebook which runs to convergence, so instead we verify
    // that the final distortion is less than the distortion of the initial
    // uniform codebook.
    let dim = 128_usize;
    let k = 8_usize;

    // Initial uniform centroids (same scheme as in the module).
    let init_centroids: Vec<f64> = (0..k)
        .map(|i| -1.0 + (2.0 * (i as f64 + 0.5)) / k as f64)
        .collect();
    let init_boundaries: Vec<f64> = init_centroids
        .windows(2)
        .map(|w| (w[0] + w[1]) / 2.0)
        .collect();

    let initial_distortion = mse_distortion(&init_centroids, &init_boundaries, dim);

    let cb = generate_codebook(3, dim);
    let final_distortion = mse_distortion(&cb.centroids, &cb.boundaries, dim);

    assert!(
        final_distortion < initial_distortion,
        "final distortion ({final_distortion}) should be < initial ({initial_distortion})"
    );
    // Distortion must be positive.
    assert!(final_distortion > 0.0);
}

/// Compute MSE distortion for test purposes.
fn mse_distortion(centroids: &[f64], boundaries: &[f64], d: usize) -> f64 {
    let k = centroids.len();
    let n = 1024_usize;
    let mut total = 0.0;
    for i in 0..k {
        let lo = if i == 0 { -1.0 } else { boundaries[i - 1] };
        let hi = if i == k - 1 { 1.0 } else { boundaries[i] };
        let c = centroids[i];
        let h = (hi - lo) / n as f64;
        let mut sum = {
            let fa = (lo - c).powi(2) * beta_pdf(lo, d);
            let fb = (hi - c).powi(2) * beta_pdf(hi, d);
            fa + fb
        };
        for j in 1..n {
            let x = lo + j as f64 * h;
            let w = if j % 2 == 0 { 2.0 } else { 4.0 };
            sum += w * (x - c).powi(2) * beta_pdf(x, d);
        }
        total += sum * h / 3.0;
    }
    total
}

// ---------------------------------------------------------------------------
// Different dimensions: wider spread for lower d
// ---------------------------------------------------------------------------

#[test]
fn higher_dim_yields_narrower_centroids() {
    let cb64 = get_codebook(3, 64).unwrap();
    let cb128 = get_codebook(3, 128).unwrap();
    let cb256 = get_codebook(3, 256).unwrap();

    // The outermost centroid should decrease as dimension grows (distribution
    // concentrates around 0).
    let outer64 = cb64.centroids.last().unwrap();
    let outer128 = cb128.centroids.last().unwrap();
    let outer256 = cb256.centroids.last().unwrap();

    assert!(
        outer64 > outer128,
        "d=64 outer ({outer64}) > d=128 ({outer128})"
    );
    assert!(
        outer128 > outer256,
        "d=128 outer ({outer128}) > d=256 ({outer256})"
    );
}

// ---------------------------------------------------------------------------
// nearest_centroid returns correct index
// ---------------------------------------------------------------------------

#[test]
fn nearest_centroid_exact_match() {
    let cb = get_codebook(3, 128).unwrap();
    for (i, &c) in cb.centroids.iter().enumerate() {
        assert_eq!(nearest_centroid(c, &cb), i as u8);
    }
}

#[test]
fn nearest_centroid_boundaries() {
    let cb = get_codebook(3, 64).unwrap();
    // A value just below a boundary should map to the lower bin.
    for (i, &b) in cb.boundaries.iter().enumerate() {
        assert_eq!(nearest_centroid(b - 1e-10, &cb), i as u8);
        // At or just above the boundary maps to the upper bin.
        assert_eq!(nearest_centroid(b + 1e-10, &cb), (i + 1) as u8);
    }
}

#[test]
fn nearest_centroid_extreme_values() {
    let cb = get_codebook(4, 256).unwrap();
    // Very negative value -> first bin.
    assert_eq!(nearest_centroid(-1.0, &cb), 0);
    // Very positive value -> last bin.
    assert_eq!(nearest_centroid(1.0, &cb), (cb.centroids.len() - 1) as u8);
}

// ---------------------------------------------------------------------------
// Convergence: generated codebook closely matches pre-computed one
// ---------------------------------------------------------------------------

#[test]
fn generated_matches_precomputed_3bit_d128() {
    let precomputed = get_codebook(3, 128).unwrap();
    let generated = generate_codebook(3, 128);

    for (pc, gc) in precomputed.centroids.iter().zip(generated.centroids.iter()) {
        assert_relative_eq!(pc, gc, epsilon = 1e-6);
    }
    for (pb, gb) in precomputed
        .boundaries
        .iter()
        .zip(generated.boundaries.iter())
    {
        assert_relative_eq!(pb, gb, epsilon = 1e-6);
    }
}

#[test]
fn generated_matches_precomputed_4bit_d64() {
    let precomputed = get_codebook(4, 64).unwrap();
    let generated = generate_codebook(4, 64);

    for (pc, gc) in precomputed.centroids.iter().zip(generated.centroids.iter()) {
        assert_relative_eq!(pc, gc, epsilon = 1e-6);
    }
}

// ---------------------------------------------------------------------------
// Beta PDF sanity
// ---------------------------------------------------------------------------

#[test]
fn beta_pdf_integrates_to_one() {
    for d in [64, 128, 256] {
        let n = 2048_usize;
        let h = 2.0 / n as f64;
        let mut sum = beta_pdf(-1.0, d) + beta_pdf(1.0, d);
        for i in 1..n {
            let x = -1.0 + i as f64 * h;
            let w = if i % 2 == 0 { 2.0 } else { 4.0 };
            sum += w * beta_pdf(x, d);
        }
        let integral = sum * h / 3.0;
        assert_relative_eq!(integral, 1.0, epsilon = 1e-4);
    }
}

#[test]
fn beta_pdf_symmetric() {
    for d in [64, 128, 256] {
        for &x in &[0.0, 0.1, 0.3, 0.5, 0.9] {
            assert_relative_eq!(beta_pdf(x, d), beta_pdf(-x, d), epsilon = 1e-12);
        }
    }
}

#[test]
fn beta_pdf_zero_outside_support() {
    // At exactly +/-1 the PDF should be 0 for d >= 3 (since (1-x^2)^((d-3)/2) = 0).
    for d in [64, 128, 256] {
        assert_relative_eq!(beta_pdf(1.0, d), 0.0, epsilon = 1e-15);
        assert_relative_eq!(beta_pdf(-1.0, d), 0.0, epsilon = 1e-15);
    }
    // Values beyond +/-1 should also be 0.
    assert_relative_eq!(beta_pdf(1.5, 128), 0.0, epsilon = 1e-15);
    assert_relative_eq!(beta_pdf(-2.0, 128), 0.0, epsilon = 1e-15);
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

#[test]
fn unsupported_bits_returns_error() {
    assert!(get_codebook(1, 64).is_err());
    assert!(get_codebook(5, 128).is_err());
}

// ---------------------------------------------------------------------------
// 2-bit codebook tests
// ---------------------------------------------------------------------------

#[test]
fn valid_2bit_codebooks() {
    for dim in [64, 128, 256] {
        let cb = get_codebook(2, dim).unwrap();
        assert_codebook_valid(&cb, 4);
    }
}

#[test]
fn symmetry_2bit_d64() {
    let cb = get_codebook(2, 64).unwrap();
    assert_symmetric(&cb);
}

#[test]
fn symmetry_2bit_d128() {
    let cb = get_codebook(2, 128).unwrap();
    assert_symmetric(&cb);
}

#[test]
fn symmetry_2bit_d256() {
    let cb = get_codebook(2, 256).unwrap();
    assert_symmetric(&cb);
}

#[test]
fn generate_2bit_d64_valid_and_symmetric() {
    let cb = generate_codebook(2, 64);
    assert_codebook_valid(&cb, 4);
    assert_symmetric(&cb);
}

#[test]
fn nearest_centroid_2bit_exact_match() {
    let cb = get_codebook(2, 128).unwrap();
    for (i, &c) in cb.centroids.iter().enumerate() {
        assert_eq!(nearest_centroid(c, &cb), i as u8);
    }
}

// ---------------------------------------------------------------------------
// Fallback: non-precomputed dim still works
// ---------------------------------------------------------------------------

#[test]
fn non_precomputed_dim_generates_on_the_fly() {
    let cb = get_codebook(3, 512).unwrap();
    assert_codebook_valid(&cb, 8);
    assert_symmetric(&cb);
}
