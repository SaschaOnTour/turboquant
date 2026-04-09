fn main() {
    #[cfg(feature = "cuda")]
    {
        use std::path::PathBuf;
        println!("cargo:rerun-if-changed=build.rs");
        println!("cargo:rerun-if-changed=src/cache/cuda/kernels/tq_common.h");
        println!("cargo:rerun-if-changed=src/cache/cuda/kernels/tq_dequant_kernel.cu");
        println!("cargo:rerun-if-changed=src/cache/cuda/kernels/tq_quant_kernel.cu");
        println!("cargo:rerun-if-changed=src/cache/cuda/kernels/tq_attention_kernel.cu");

        let builder = cudaforge::KernelBuilder::new()
            .source_glob("src/cache/cuda/kernels/*.cu")
            .arg("-std=c++17")
            .arg("-O3")
            .arg("-U__CUDA_NO_HALF_OPERATORS__")
            .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
            .arg("-U__CUDA_NO_HALF2_OPERATORS__")
            .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
            .arg("--expt-relaxed-constexpr")
            .arg("--expt-extended-lambda")
            .arg("--use_fast_math")
            .arg("--compiler-options")
            .arg("-fPIC");

        let build_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
        let target = std::env::var("TARGET").unwrap();
        let out_file = if target.contains("msvc") {
            build_dir.join("turboquant_cuda.lib")
        } else {
            build_dir.join("libturboquant_cuda.a")
        };

        builder
            .build_lib(out_file)
            .expect("Failed to build TurboQuant CUDA kernels");

        println!("cargo:rustc-link-search={}", build_dir.display());
        println!("cargo:rustc-link-lib=turboquant_cuda");
        println!("cargo:rustc-link-lib=dylib=cudart");
    }
}
