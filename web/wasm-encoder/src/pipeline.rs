use crate::color::{self, Mat3};
use crate::encode_avif;
use crate::encode_jpeg;
use crate::encode_tiff;
use crate::encode_uhdr;
use crate::rotation::{self, Orientation};
use crate::transfer;

#[derive(Clone, Copy)]
pub enum Format {
    Avif,
    Jpeg,
    JpegHdr,
    Tiff,
}

impl Format {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "avif" => Ok(Format::Avif),
            "jpeg" => Ok(Format::Jpeg),
            "jpeg-hdr" => Ok(Format::JpegHdr),
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
    wb: &[f32; 3],
    orientation: &str,
    format: Format,
    quality: u8,
) -> Result<Vec<u8>, String> {
    // Check if CC is already applied (zero matrix = pre-corrected sRGB data)
    let cc_applied = xyz_to_cam.iter().all(|row| row.iter().all(|v| v.abs() < 1e-10));

    // Select color matrix and highlight-blend simple matrix based on format
    let (color_matrix, simple_matrix) = if cc_applied {
        // Data is already in sRGB linear — just apply gamut conversion if needed
        match format {
            Format::Avif => (color::SRGB_TO_BT2020, color::SRGB_TO_BT2020),
            Format::Jpeg | Format::JpegHdr | Format::Tiff => (color::IDENTITY, color::IDENTITY),
        }
    } else {
        match format {
            Format::Avif => (
                color::build_cam_to_bt2020(xyz_to_cam),
                color::SRGB_TO_BT2020,
            ),
            Format::Jpeg | Format::JpegHdr | Format::Tiff => (color::build_cam_to_srgb(xyz_to_cam), color::IDENTITY),
        }
    };

    let n = (width as usize) * (height as usize);
    let wb_r = wb[0].max(1e-6);
    let wb_g = wb[1].max(1e-6);
    let wb_b = wb[2].max(1e-6);

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

        // Highlight blend: ramp from full correction to simple near sensor clipping.
        // Divide WB'd values by WB multiplier to get sensor-space proximity [0..1].
        let clip_prox = (r / wb_r).max(g / wb_g).max(b / wb_b);
        let alpha = ((clip_prox - 0.85) / 0.15).clamp(0.0, 1.0);

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
            // Scale super-whites into [0,1] instead of hard-clipping
            let peak = rotated.iter().cloned().fold(0.0f32, f32::max);
            let scale = if peak > 1.0 { 1.0 / peak } else { 1.0 };
            // sRGB gamma → 8-bit
            let mut rgb8 = vec![0u8; num * 3];
            for i in 0..num * 3 {
                rgb8[i] = (transfer::srgb_oetf(rotated[i] * scale) * 255.0 + 0.5) as u8;
            }
            encode_jpeg::encode(&rgb8, rw, rh, quality)
        }

        Format::JpegHdr => {
            // Ultra HDR JPEG: SDR base + gain map for highlight recovery
            let peak = rotated.iter().cloned().fold(0.0f32, f32::max);
            let scale = if peak > 1.0 { 1.0 / peak } else { 1.0 };

            // No HDR headroom — fall back to regular JPEG
            if peak <= 1.0 {
                let mut rgb8 = vec![0u8; num * 3];
                for i in 0..num * 3 {
                    rgb8[i] = (transfer::srgb_oetf(rotated[i]) * 255.0 + 0.5) as u8;
                }
                return encode_jpeg::encode(&rgb8, rw, rh, quality);
            }

            // SDR base: peak-scaled sRGB
            let mut sdr_rgb8 = vec![0u8; num * 3];
            for i in 0..num * 3 {
                sdr_rgb8[i] = (transfer::srgb_oetf(rotated[i] * scale) * 255.0 + 0.5) as u8;
            }

            // Gain map: per-pixel luminance log2 ratio
            let offset: f32 = 1.0 / 64.0;
            let mut gains = vec![0.0f32; num];
            let mut gain_min: f32 = f32::MAX;
            let mut gain_max: f32 = f32::MIN;

            for i in 0..num {
                let idx = i * 3;
                let y_hdr = 0.2126 * rotated[idx] + 0.7152 * rotated[idx + 1] + 0.0722 * rotated[idx + 2];
                let y_sdr = y_hdr * scale;
                let gain = ((y_hdr + offset) / (y_sdr + offset)).max(1e-10).log2();
                gains[i] = gain;
                gain_min = gain_min.min(gain);
                gain_max = gain_max.max(gain);
            }

            // Normalize to [0, 255]
            let range = (gain_max - gain_min).max(1e-6);
            let mut gain_luma8 = vec![0u8; num];
            for i in 0..num {
                gain_luma8[i] = ((gains[i] - gain_min) / range * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            }

            encode_uhdr::encode(
                &sdr_rgb8, rw, rh, quality,
                &gain_luma8,
                gain_min, gain_max, offset,
            )
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
            // HLG OETF → find peak → normalize → 10-bit
            let mut hlg_buf = vec![0.0f32; num * 3];
            let mut peak: f32 = 0.0;
            for i in 0..num * 3 {
                let v = transfer::hlg_oetf(rotated[i]);
                hlg_buf[i] = v;
                if v > peak {
                    peak = v;
                }
            }
            let scale = if peak > 1.0 { 1.0 / peak } else { 1.0 };
            let mut rgb10 = vec![0u16; num * 3];
            for i in 0..num * 3 {
                rgb10[i] = (hlg_buf[i] * scale * 1023.0 + 0.5).clamp(0.0, 1023.0) as u16;
            }
            encode_avif::encode(&rgb10, rw, rh, quality)
        }
    }
}
