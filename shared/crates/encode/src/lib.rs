use wasm_bindgen::prelude::*;

mod encode_avif;
mod encode_jpeg;
mod encode_tiff;
mod encode_uhdr;
mod exif;
mod pipeline;
mod rotation;
mod transfer;

#[wasm_bindgen]
pub fn encode_image(
    data: &[f32],
    hdr_data: &[f32],
    width: u32,
    height: u32,
    orientation: &str,
    format: &str,
    quality: u8,
    peak_luminance: f32,
) -> Result<Vec<u8>, JsError> {
    console_error_panic_hook::set_once();

    let pixels = (width as usize) * (height as usize) * 3;
    if data.len() != pixels {
        return Err(JsError::new("data length mismatch: expected width * height * 3"));
    }

    let fmt = pipeline::Format::parse(format).map_err(|e| JsError::new(&e))?;

    if matches!(fmt, pipeline::Format::JpegHdr) && hdr_data.len() != pixels {
        return Err(JsError::new("hdr_data length mismatch for jpeg-hdr: expected width * height * 3"));
    }

    pipeline::encode(data, hdr_data, width, height, orientation, fmt, quality, peak_luminance)
        .map_err(|e| JsError::new(&e))
}
