use wasm_bindgen::prelude::*;
use xveon_encode::{encode, EncodeOptions, Format};
#[wasm_bindgen]
pub fn encode_image(data: &[f32], hdr_data: &[f32], width: u32, height: u32, orientation: &str,
                    format: &str, quality: u8, peak_luminance: f32) -> Result<Vec<u8>, JsError> {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
    let format = Format::parse(format).map_err(|e| JsError::new(&e))?;
    encode(data, hdr_data, width, height, orientation, format, quality, peak_luminance, EncodeOptions { threads: 1 })
        .map_err(|e| JsError::new(&e))
}
