use std::io::Cursor;
use tiff::encoder::{colortype::RGB16, TiffEncoder};

pub fn encode(rgb16: &[u16], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let mut buf = Cursor::new(Vec::new());
    {
        let mut encoder =
            TiffEncoder::new(&mut buf).map_err(|e| format!("TIFF init error: {e}"))?;
        encoder
            .write_image::<RGB16>(width, height, rgb16)
            .map_err(|e| format!("TIFF write error: {e}"))?;
    }
    Ok(buf.into_inner())
}
