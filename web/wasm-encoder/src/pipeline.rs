use crate::color::{self, Mat3};
use crate::encode_avif;
use crate::encode_jpeg;
use crate::encode_tiff;
use crate::rotation::{self, Orientation};
use crate::transfer;

#[derive(Clone, Copy)]
pub enum Format {
    Avif,
    Jpeg,
    Tiff,
}

impl Format {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "avif" => Ok(Format::Avif),
            "jpeg" => Ok(Format::Jpeg),
            "tiff" => Ok(Format::Tiff),
            _ => Err(format!("unknown format: {s}")),
        }
    }
}

pub fn encode(
    hwc: &[f32],
    width: u32,
    height: u32,
    xyz_to_cam: &Mat3,
    orientation: &str,
    format: Format,
    quality: u8,
) -> Result<Vec<u8>, String> {
    // Select color matrix and highlight-blend simple matrix based on format
    let (color_matrix, simple_matrix) = match format {
        Format::Avif => (
            color::build_cam_to_bt2020(xyz_to_cam),
            color::SRGB_TO_BT2020,
        ),
        Format::Jpeg | Format::Tiff => (color::build_cam_to_srgb(xyz_to_cam), color::IDENTITY),
    };

    let n = (width as usize) * (height as usize);

    // Color correction + highlight blending
    let mut buf = Vec::with_capacity(n * 3);
    for i in 0..n {
        let idx = i * 3;
        let r = hwc[idx];
        let g = hwc[idx + 1];
        let b = hwc[idx + 2];

        // Full color matrix
        let fr = color_matrix[0][0] * r + color_matrix[0][1] * g + color_matrix[0][2] * b;
        let fg = color_matrix[1][0] * r + color_matrix[1][1] * g + color_matrix[1][2] * b;
        let fb = color_matrix[2][0] * r + color_matrix[2][1] * g + color_matrix[2][2] * b;

        // Simple matrix (identity for sRGB, SRGB_TO_BT2020 for AVIF)
        let sr = simple_matrix[0][0] * r + simple_matrix[0][1] * g + simple_matrix[0][2] * b;
        let sg = simple_matrix[1][0] * r + simple_matrix[1][1] * g + simple_matrix[1][2] * b;
        let sb = simple_matrix[2][0] * r + simple_matrix[2][1] * g + simple_matrix[2][2] * b;

        // Highlight blend: ramp from full correction to simple for bright pixels
        let max_ch = r.max(g).max(b);
        let alpha = ((max_ch - 0.8) / 0.7).clamp(0.0, 1.0);

        buf.push((fr + alpha * (sr - fr)).max(0.0));
        buf.push((fg + alpha * (sg - fg)).max(0.0));
        buf.push((fb + alpha * (sb - fb)).max(0.0));
    }

    // EXIF rotation
    let orient = Orientation::parse(orientation);
    let (rotated, rw, rh) = rotation::apply_rotation(&buf, width, height, &orient);

    let num = (rw as usize) * (rh as usize);

    match format {
        Format::Jpeg => {
            // sRGB gamma → 8-bit
            let mut rgb8 = vec![0u8; num * 3];
            for i in 0..num * 3 {
                rgb8[i] = (transfer::srgb_oetf(rotated[i]) * 255.0 + 0.5) as u8;
            }
            encode_jpeg::encode(&rgb8, rw, rh, quality)
        }

        Format::Tiff => {
            // Linear, clamp [0,1] → 16-bit
            let mut rgb16 = vec![0u16; num * 3];
            for i in 0..num * 3 {
                rgb16[i] = (rotated[i].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
            }
            encode_tiff::encode(&rgb16, rw, rh)
        }

        Format::Avif => {
            // HLG OETF → 10-bit
            let mut rgb10 = vec![0u16; num * 3];
            for i in 0..num * 3 {
                rgb10[i] = (transfer::hlg_oetf(rotated[i]) * 1023.0 + 0.5).clamp(0.0, 1023.0)
                    as u16;
            }
            encode_avif::encode(&rgb10, rw, rh, quality)
        }
    }
}
