mod par;
mod encode_avif; mod encode_jpeg; mod encode_tiff; mod encode_uhdr; mod exif; mod math; mod pipeline; mod rotation; mod transfer;
pub use pipeline::Format;
/// The only thing a wrapper chooses. The WASM build passes 1; the native build passes the
/// machine's parallelism. The output never depends on it.
#[derive(Clone, Copy, Debug)]
pub struct EncodeOptions { pub threads: usize }
/// Display-linear HWC float pixels in, file bytes out (see pipeline::encode for the planes per format).
pub fn encode(data: &[f32], hdr_data: &[f32], width: u32, height: u32, orientation: &str, format: Format,
              quality: u8, peak_luminance: f32, options: EncodeOptions) -> Result<Vec<u8>, String> {
    if width == 0 || height == 0 { return Err("empty image".into()); }
    let pixels = (width as usize).checked_mul(height as usize).and_then(|n| n.checked_mul(3)).ok_or("image too large")?;
    if data.len() != pixels { return Err("data length mismatch: expected width * height * 3".into()); }
    if matches!(format, Format::JpegHdr) {
        if hdr_data.len() != pixels { return Err("hdr_data length mismatch for jpeg-hdr: expected width * height * 3".into()); }
        if width > u16::MAX as u32 || height > u16::MAX as u32 { return Err("JPEG images are limited to 65535 pixels per side".into()); }
    }
    pipeline::encode(data, hdr_data, width, height, orientation, format, quality, peak_luminance, options)
}
