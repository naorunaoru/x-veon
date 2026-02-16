use jpeg_encoder::{ColorType, Encoder};

pub fn encode(rgb8: &[u8], width: u32, height: u32, quality: u8) -> Result<Vec<u8>, String> {
    let mut buf = Vec::new();
    let encoder = Encoder::new(&mut buf, quality);
    encoder
        .encode(rgb8, width as u16, height as u16, ColorType::Rgb)
        .map_err(|e| format!("JPEG encode error: {e}"))?;
    Ok(buf)
}

