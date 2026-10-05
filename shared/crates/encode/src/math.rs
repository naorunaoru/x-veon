//! Transcendental functions from the `libm` crate on every target. std's f32 methods call the
//! platform libm on native targets but Rust's bundled libm on wasm32, and their last bits can
//! differ; the WASM and native builds must write identical files (spec §7, §11).
#[inline] pub fn powf(x: f32, y: f32) -> f32 { libm::powf(x, y) }
#[inline] pub fn logf(x: f32) -> f32 { libm::logf(x) }
#[inline] pub fn log2f(x: f32) -> f32 { libm::log2f(x) }
#[inline] pub fn sqrtf(x: f32) -> f32 { libm::sqrtf(x) }
