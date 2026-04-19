//! TurboQuantConfig validation tests.
//!
//! Extracted from `packed_tests.rs`.

use turboquant::packed::TurboQuantConfig;

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

#[test]
fn config_accepts_valid_2bit() {
    // Validates that new(2, 64) succeeds -- the config is usable for quantization.
    let _cfg = TurboQuantConfig::new(2, 64).unwrap();
}
