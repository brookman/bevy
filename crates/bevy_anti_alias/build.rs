//! Compiles the Objective-C MetalFX shim when the `metalfx` feature is enabled on macOS.

fn main() {
    println!("cargo:rerun-if-changed=src/metalfx/shim.m");
    #[cfg(feature = "metalfx")]
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        cc::Build::new()
            .file("src/metalfx/shim.m")
            .flag("-fobjc-arc")
            .compile("bevy_metalfx_shim");
        println!("cargo:rustc-link-lib=framework=MetalFX");
    }
}
