fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    // Check if openblas feature is enabled via CARGO_FEATURE_OPENBLAS env var
    if std::env::var("CARGO_FEATURE_OPENBLAS").is_ok() {
        match target_os.as_str() {
            "macos" => {
                println!("cargo:rustc-link-lib=framework=Accelerate");
            }
            "linux" => {
                println!("cargo:rustc-link-lib=openblas");
            }
            _ => {
                // Default fallback linkage
                println!("cargo:rustc-link-lib=openblas");
            }
        }
    }
}
