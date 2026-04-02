//! Mathematical helper functions used across TurboQuant modules.
//!
//! Contains the Lanczos gamma approximation and Simpson's rule numerical
//! integration, extracted from [`codebook`] for single-responsibility.

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Lanczos approximation parameter g.
pub(crate) const LANCZOS_G: f64 = 7.0;

/// Reflection formula threshold for the Lanczos gamma approximation.
pub(crate) const LANCZOS_REFLECTION_THRESHOLD: f64 = 0.5;

/// Divisor used throughout to halve a value (d/2, (d-1)/2, 0.5*ln(pi), etc.).
pub(crate) const HALF: f64 = 0.5;

/// Simpson's rule weight for even-indexed interior points.
pub(crate) const SIMPSON_WEIGHT_EVEN: f64 = 2.0;

/// Simpson's rule weight for odd-indexed interior points.
pub(crate) const SIMPSON_WEIGHT_ODD: f64 = 4.0;

/// Simpson's rule divisor (h / 3).
pub(crate) const SIMPSON_DIVISOR: f64 = 3.0;

/// Lanczos coefficients for g = 7, n = 9.
pub(crate) const LANCZOS_COEFFS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];

// ---------------------------------------------------------------------------
// Pure Operation: Lanczos ln(Gamma)
// ---------------------------------------------------------------------------

/// Natural log of the Gamma function via the Lanczos approximation.
///
/// Uses the reflection formula for `z < 0.5` and the standard Lanczos
/// series otherwise.
///
/// Pure Operation: arithmetic only.
pub fn ln_gamma(z: f64) -> f64 {
    let (use_reflection, zd) = if z < LANCZOS_REFLECTION_THRESHOLD {
        (true, 1.0 - z)
    } else {
        (false, z)
    };
    let zd_minus_one = zd - 1.0;
    let mut x = LANCZOS_COEFFS[0];
    for (i, &c) in LANCZOS_COEFFS.iter().enumerate().skip(1) {
        x += c / (zd_minus_one + i as f64);
    }
    let t = zd_minus_one + LANCZOS_G + HALF;
    let ln_two_pi_half = HALF * core::f64::consts::TAU.ln();
    let direct = ln_two_pi_half + (t.ln() * (zd_minus_one + HALF)) - t + x.ln();
    if use_reflection {
        let pi = core::f64::consts::PI;
        pi.ln() - (pi * z).sin().abs().ln() - direct
    } else {
        direct
    }
}

// ---------------------------------------------------------------------------
// Pure Operation: Simpson's rule integration
// ---------------------------------------------------------------------------

/// Simpson's rule numerical integration of `f` over `[a, b]`.
///
/// Uses `steps` sub-intervals (must be even).
///
/// Pure Operation: performs only arithmetic on the closure's results.
pub fn simpsons_integrate<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, steps: usize) -> f64 {
    let h = (b - a) / steps as f64;
    let mut sum = f(a) + f(b);
    for i in 1..steps {
        let x = a + i as f64 * h;
        let weight = if i % 2 == 0 {
            SIMPSON_WEIGHT_EVEN
        } else {
            SIMPSON_WEIGHT_ODD
        };
        sum += weight * f(x);
    }
    sum * h / SIMPSON_DIVISOR
}

// ---------------------------------------------------------------------------
// Pure Operation: convergence loop
// ---------------------------------------------------------------------------

/// Runs `step` up to `max_iter` times, stopping early when it returns `true`.
///
/// Pure Operation: for-loop + if-break only, no own function calls (the step
/// closure is caller-provided, not a project function).
pub fn converge<F: FnMut() -> bool>(max_iter: usize, mut step: F) {
    for _ in 0..max_iter {
        if step() {
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Constant function value used in Simpson's rule constant-function test.
    const SIMPSONS_TEST_CONST_VALUE: f64 = 5.0;

    /// Upper integration limit for Simpson's constant-function test.
    const SIMPSONS_TEST_CONST_UPPER: f64 = 2.0;

    /// Expected result of Simpson's constant-function test (value * upper).
    const SIMPSONS_TEST_CONST_EXPECTED: f64 = 10.0;

    /// Number of integration steps for tests.
    const TEST_INTEGRATION_STEPS: usize = 1024;

    // -- ln_gamma -----------------------------------------------------------

    #[test]
    fn ln_gamma_of_one_is_zero() {
        assert_relative_eq!(ln_gamma(1.0), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn ln_gamma_of_half_is_half_ln_pi() {
        let expected = HALF * core::f64::consts::PI.ln();
        assert_relative_eq!(ln_gamma(0.5), expected, epsilon = 1e-10);
    }

    #[test]
    fn ln_gamma_of_two_is_zero() {
        assert_relative_eq!(ln_gamma(2.0), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn ln_gamma_of_five_is_ln_24() {
        assert_relative_eq!(ln_gamma(5.0), 24.0_f64.ln(), epsilon = 1e-8);
    }

    #[test]
    fn ln_gamma_known_values_across_range() {
        // Reference values: ln(Gamma(z)) for known inputs.
        // Gamma(0.1) ≈ 9.51351..., ln ≈ 2.25271...
        // Gamma(2.5) = 1.5*0.5*Gamma(0.5) = 0.75*sqrt(pi) ≈ 1.32934...
        // Gamma(10) = 9! = 362880, ln ≈ 12.80183...
        const EXPECTED: [(f64, f64); 3] = [
            (0.1, 2.252_712_651_734_206),
            (2.5, 0.284_682_870_472_919_2),
            (10.0, 12.801_827_480_081_469),
        ];
        for &(z, expected) in &EXPECTED {
            assert_relative_eq!(ln_gamma(z), expected, epsilon = 1e-8);
        }
    }

    // -- simpsons_integrate -------------------------------------------------

    #[test]
    fn simpsons_x_squared_zero_to_one() {
        let result = simpsons_integrate(|x| x * x, 0.0, 1.0, TEST_INTEGRATION_STEPS);
        assert_relative_eq!(result, 1.0 / 3.0, epsilon = 1e-8);
    }

    #[test]
    fn simpsons_constant_function() {
        let result = simpsons_integrate(
            |_| SIMPSONS_TEST_CONST_VALUE,
            0.0,
            SIMPSONS_TEST_CONST_UPPER,
            TEST_INTEGRATION_STEPS,
        );
        assert_relative_eq!(result, SIMPSONS_TEST_CONST_EXPECTED, epsilon = 1e-10);
    }

    #[test]
    fn simpsons_sin_zero_to_pi() {
        let result = simpsons_integrate(
            |x| x.sin(),
            0.0,
            core::f64::consts::PI,
            TEST_INTEGRATION_STEPS,
        );
        assert_relative_eq!(result, 2.0, epsilon = 1e-8);
    }
}
