use wasm_bindgen::prelude::*;

/// Demosaic a single-channel CFA image to 3-channel planar RGB.
///
/// Input: f32 CFA data (normalized, WB'd), row-major, length = width * height.
/// Output: f32 planar CHW (R plane, G plane, B plane), length = 3 * width * height.
///
/// `dy`, `dx`: CFA pattern shift (additive convention).
/// `algorithm`: "bilinear" or "markesteijn1".
#[wasm_bindgen]
pub fn demosaic_image(
    cfa: &[f32],
    width: u32,
    height: u32,
    dy: u32,
    dx: u32,
    algorithm: &str,
) -> Result<Vec<f32>, JsError> {
    console_error_panic_hook::set_once();

    let w = width as usize;
    let h = height as usize;

    if cfa.len() != w * h {
        return Err(JsError::new(&format!(
            "CFA length mismatch: expected {}, got {}",
            w * h,
            cfa.len()
        )));
    }

    let pattern = demosaic::CfaPattern::xtrans_default().shift(dy as usize, dx as usize);

    let algo = match algorithm {
        "bilinear" => demosaic::Algorithm::Bilinear,
        "markesteijn1" => demosaic::Algorithm::Markesteijn1,
        "markesteijn3" => demosaic::Algorithm::Markesteijn3,
        "dht" => demosaic::Algorithm::Dht,
        _ => return Err(JsError::new(&format!("unknown algorithm: '{algorithm}'"))),
    };

    let mut output = vec![0.0f32; 3 * w * h];
    demosaic::demosaic(cfa, w, h, &pattern, algo, &mut output)
        .map_err(|e| JsError::new(&format!("{e}")))?;

    Ok(output)
}

/// Demosaic a Bayer CFA image to 3-channel planar RGB.
///
/// `cfa_type`: "rggb", "bggr", "grbg", or "gbrg".
#[wasm_bindgen]
pub fn demosaic_bayer(
    cfa: &[f32],
    width: u32,
    height: u32,
    cfa_type: &str,
) -> Result<Vec<f32>, JsError> {
    console_error_panic_hook::set_once();

    let w = width as usize;
    let h = height as usize;

    if cfa.len() != w * h {
        return Err(JsError::new(&format!(
            "CFA length mismatch: expected {}, got {}",
            w * h,
            cfa.len()
        )));
    }

    let pattern = match cfa_type {
        "rggb" => demosaic::CfaPattern::bayer_rggb(),
        "bggr" => demosaic::CfaPattern::bayer_bggr(),
        "grbg" => demosaic::CfaPattern::bayer_grbg(),
        "gbrg" => demosaic::CfaPattern::bayer_gbrg(),
        _ => return Err(JsError::new(&format!("unknown Bayer pattern: '{cfa_type}'"))),
    };

    let mut output = vec![0.0f32; 3 * w * h];
    demosaic::demosaic(cfa, w, h, &pattern, demosaic::Algorithm::Bilinear, &mut output)
        .map_err(|e| JsError::new(&format!("{e}")))?;

    Ok(output)
}
