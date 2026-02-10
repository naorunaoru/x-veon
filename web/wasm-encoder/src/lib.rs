use wasm_bindgen::prelude::*;

mod color;
mod encode_avif;
mod encode_jpeg;
mod encode_tiff;
mod pipeline;
mod rotation;
mod transfer;

#[wasm_bindgen]
pub fn encode_image(
    data: &[f32],
    width: u32,
    height: u32,
    xyz_to_cam: &[f32],
    orientation: &str,
    format: &str,
    quality: u8,
) -> Result<Vec<u8>, JsError> {
    console_error_panic_hook::set_once();

    if data.len() != (width as usize) * (height as usize) * 3 {
        return Err(JsError::new("data length mismatch: expected width * height * 3"));
    }
    if xyz_to_cam.len() < 9 {
        return Err(JsError::new("xyz_to_cam must have at least 9 elements"));
    }

    let mat = color::mat3_from_slice(xyz_to_cam);
    let fmt = pipeline::Format::parse(format).map_err(|e| JsError::new(&e))?;

    pipeline::encode(data, width, height, &mat, orientation, fmt, quality)
        .map_err(|e| JsError::new(&e))
}
