use approx::assert_relative_eq;
use turboquant::codebook::{beta_pdf, generate_codebook, get_codebook, nearest_centroid, Codebook};

/// Supported bit widths paired with their expected codebook cardinality `k = 2^bits`.
const BITS_K: &[(u8, usize)] = &[(2, 4), (3, 8), (4, 16)];

/// Standard head dimensions exercised by the pre-computed + generated codebook tables.
const DIMS: &[usize] = &[64, 128, 256];

fn assert_codebook_valid(cb: &Codebook, expected_k: usize) {
    assert_eq!(cb.centroids.len(), expected_k);
    assert_eq!(cb.boundaries.len(), expected_k - 1);

    for &c in &cb.centroids {
        assert!((-1.0..=1.0).contains(&c), "centroid {c} outside [-1, 1]");
    }

    for w in cb.centroids.windows(2) {
        assert!(w[0] < w[1], "centroids not sorted: {} >= {}", w[0], w[1]);
    }

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
    if m % 2 == 1 {
        assert_relative_eq!(cb.boundaries[m / 2], 0.0, epsilon = 1e-10);
    }
}

#[test]
fn precomputed_codebooks_are_valid_and_symmetric() {
    for &(bits, expected_k) in BITS_K {
        for &dim in DIMS {
            let cb = get_codebook(bits, dim)
                .unwrap_or_else(|_| panic!("get_codebook({bits}, {dim}) failed"));
            assert_codebook_valid(&cb, expected_k);
            assert_symmetric(&cb);
        }
    }
}

#[test]
fn generated_codebooks_are_valid_and_symmetric() {
    for &(bits, expected_k) in BITS_K {
        for &dim in DIMS {
            let cb = generate_codebook(bits, dim);
            assert_codebook_valid(&cb, expected_k);
            assert_symmetric(&cb);
        }
    }
}

#[test]
fn generated_matches_precomputed() {
    for &(bits, _k) in BITS_K {
        for &dim in DIMS {
            let precomputed = get_codebook(bits, dim).unwrap();
            let generated = generate_codebook(bits, dim);
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
    }
}

#[test]
fn distortion_decreases_over_iterations() {
    let dim = 128_usize;
    let k = 8_usize;

    let init_centroids: Vec<f64> = (0..k)
        .map(|i| -1.0 + (2.0 * (i as f64 + 0.5)) / k as f64)
        .collect();
    let init_boundaries: Vec<f64> = init_centroids
        .windows(2)
        .map(|w| (w[0] + w[1]) / 2.0)
        .collect();

    let initial_distortion = mse_distortion(&init_centroids, &init_boundaries, dim);

    let cb = generate_codebook(3, dim);
    assert_codebook_valid(&cb, k);
    assert_symmetric(&cb);
    let final_distortion = mse_distortion(&cb.centroids, &cb.boundaries, dim);

    assert!(
        final_distortion < initial_distortion,
        "final distortion ({final_distortion}) should be < initial ({initial_distortion})"
    );
    assert!(final_distortion > 0.0);
}

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

#[test]
fn higher_dim_yields_narrower_centroids() {
    let cb64 = get_codebook(3, 64).unwrap();
    let cb128 = get_codebook(3, 128).unwrap();
    let cb256 = get_codebook(3, 256).unwrap();
    assert_codebook_valid(&cb64, 8);
    assert_codebook_valid(&cb128, 8);
    assert_codebook_valid(&cb256, 8);

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

#[test]
fn nearest_centroid_exact_match() {
    for &(bits, _) in BITS_K {
        let cb = get_codebook(bits, 128).unwrap();
        for (i, &c) in cb.centroids.iter().enumerate() {
            assert_eq!(nearest_centroid(c, &cb), i as u8);
        }
    }
}

#[test]
fn nearest_centroid_boundaries() {
    let cb = get_codebook(3, 64).unwrap();
    for (i, &b) in cb.boundaries.iter().enumerate() {
        assert_eq!(nearest_centroid(b - 1e-10, &cb), i as u8);
        assert_eq!(nearest_centroid(b + 1e-10, &cb), (i + 1) as u8);
    }
}

#[test]
fn nearest_centroid_extreme_values() {
    let cb = get_codebook(4, 256).unwrap();
    assert_eq!(nearest_centroid(-1.0, &cb), 0);
    assert_eq!(nearest_centroid(1.0, &cb), (cb.centroids.len() - 1) as u8);
}

/// Simpson-rule sub-intervals for integrating Beta PDF over [-1, 1].
const BETA_SIMPSON_STEPS: usize = 2048;

#[test]
fn beta_pdf_integrates_to_one() {
    for &d in DIMS {
        let n = BETA_SIMPSON_STEPS;
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
    const SYMMETRY_SAMPLE_POINTS: [f64; 5] = [0.0, 0.1, 0.3, 0.5, 0.9];
    for &d in DIMS {
        for &x in &SYMMETRY_SAMPLE_POINTS {
            assert_relative_eq!(beta_pdf(x, d), beta_pdf(-x, d), epsilon = 1e-12);
        }
    }
}

#[test]
fn beta_pdf_zero_outside_support() {
    for &d in DIMS {
        assert_relative_eq!(beta_pdf(1.0, d), 0.0, epsilon = 1e-15);
        assert_relative_eq!(beta_pdf(-1.0, d), 0.0, epsilon = 1e-15);
    }
    assert_relative_eq!(beta_pdf(1.5, 128), 0.0, epsilon = 1e-15);
    assert_relative_eq!(beta_pdf(-2.0, 128), 0.0, epsilon = 1e-15);
}

#[test]
fn unsupported_bits_returns_error() {
    assert!(get_codebook(1, 64).is_err());
    assert!(get_codebook(5, 128).is_err());
}

#[test]
fn non_precomputed_dim_generates_on_the_fly() {
    let cb = get_codebook(3, 512).unwrap();
    assert_codebook_valid(&cb, 8);
    assert_symmetric(&cb);
}
