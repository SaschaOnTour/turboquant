//! Basic TurboQuant usage: quantize and dequantize a single vector.
//!
//! Demonstrates the core PolarQuant pipeline:
//! 1. Create a configuration (3-bit, d=128)
//! 2. Quantize a vector
//! 3. Dequantize it back
//! 4. Measure compression ratio and reconstruction error
//!
//! No external dependencies (no `rand`). Uses a simple sine pattern.

use turboquant::packed::TurboQuantConfig;
use turboquant::quantize::{dequantize_vec, quantize_vec};

// ---------------------------------------------------------------------------
// Named constants
// ---------------------------------------------------------------------------

/// Number of quantization bits (TQ3).
const BITS: u8 = 3;

/// Head dimension (typical for LLMs like LLaMA, Mistral).
const DIM: usize = 128;

/// Seed for the rotation sign pattern (any fixed value for reproducibility).
const ROTATION_SEED: u64 = 42;

/// Size of a single FP32 element in bytes.
const BYTES_PER_F32: usize = 4;

/// Frequency scaling for the sine input pattern.
const SINE_FREQUENCY: f64 = 0.1;

/// Amplitude scaling for the sine input pattern.
const SINE_AMPLITUDE: f32 = 2.5;

fn main() {
    // -- 1. Configure ---------------------------------------------------------
    let config = TurboQuantConfig::new(BITS, DIM)
        .expect("valid config")
        .with_seed(ROTATION_SEED);

    // -- 2. Generate a deterministic input vector (sine pattern) ---------------
    let input: Vec<f32> = (0..DIM)
        .map(|i| SINE_AMPLITUDE * ((i as f64 * SINE_FREQUENCY).sin() as f32))
        .collect();

    let original_bytes = DIM * BYTES_PER_F32;

    // -- 3. Quantize ----------------------------------------------------------
    let block = quantize_vec(&config, &input).expect("quantization succeeded");
    let packed_bytes = block.size_bytes();

    // -- 4. Dequantize --------------------------------------------------------
    let recovered = dequantize_vec(&config, &block).expect("dequantization succeeded");

    // -- 5. Compute metrics ---------------------------------------------------
    let mse: f32 = input
        .iter()
        .zip(recovered.iter())
        .map(|(&a, &b)| {
            let diff = a - b;
            diff * diff
        })
        .sum::<f32>()
        / DIM as f32;

    let input_energy: f32 = input.iter().map(|&x| x * x).sum::<f32>() / DIM as f32;
    let normalized_mse = mse / input_energy;
    let compression_ratio = original_bytes as f64 / packed_bytes as f64;

    // -- 6. Print results -----------------------------------------------------
    println!("TurboQuant TQ{BITS} basic example (d={DIM})");
    println!("---------------------------------------------");
    println!("Original size:      {original_bytes} bytes (FP32)");
    println!("Packed size:        {packed_bytes} bytes");
    println!("Compression ratio:  {compression_ratio:.1}x");
    println!("MSE:                {mse:.6}");
    println!("Normalized MSE:     {normalized_mse:.6}");
    println!();
    println!("First 8 values:");
    println!("  original:   {:?}", &input[..8]);
    println!("  recovered:  {:?}", &recovered[..8]);
}
