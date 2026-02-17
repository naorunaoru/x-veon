use crate::encode_avif;
use crate::encode_tiff;
use crate::encode_uhdr;
use crate::exif;
use crate::rotation::{self, Orientation};
use crate::transfer;

#[derive(Clone, Copy)]
pub enum Format {
    Avif,
    JpegHdr,
    Tiff,
}

impl Format {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "avif" => Ok(Format::Avif),
            "jpeg-hdr" => Ok(Format::JpegHdr),
            "tiff" => Ok(Format::Tiff),
            _ => Err(format!("unknown format: {s}")),
        }
    }
}

/// Encode display-linear tonemapped data into the requested format.
///
/// - `data`: display-linear Rec.709 (JPEG/TIFF) or Rec.2020 (AVIF) pixels, HWC layout
/// - `hdr_data`: display-linear Rec.2020 HDR pixels (only used for JPEG-HDR)
/// - `orientation`: EXIF orientation string — embedded as metadata for JPEG/JPEG-HDR,
///    applied as physical rotation for TIFF/AVIF
/// - `peak_luminance`: HDR peak in nits (e.g. 1000.0), used for gain map computation
pub fn encode(
    data: &[f32],
    hdr_data: &[f32],
    width: u32,
    height: u32,
    orientation: &str,
    format: Format,
    quality: u8,
    peak_luminance: f32,
) -> Result<Vec<u8>, String> {
    match format {
        Format::JpegHdr => {
            // No physical rotation — EXIF orientation embedded in primary JPEG
            let num = (width as usize) * (height as usize);

            // SDR: apply sRGB OETF, quantize to u8
            let mut sdr_rgb8 = vec![0u8; num * 3];
            for i in 0..num {
                let idx = i * 3;
                sdr_rgb8[idx]     = (transfer::srgb_oetf(data[idx].clamp(0.0, 1.0)) * 255.0 + 0.5) as u8;
                sdr_rgb8[idx + 1] = (transfer::srgb_oetf(data[idx + 1].clamp(0.0, 1.0)) * 255.0 + 0.5) as u8;
                sdr_rgb8[idx + 2] = (transfer::srgb_oetf(data[idx + 2].clamp(0.0, 1.0)) * 255.0 + 0.5) as u8;
            }

            // Compute gain map from display-linear SDR and HDR values
            let sdr_peak: f32 = 100.0;
            let peak_ratio = peak_luminance / sdr_peak;
            let offset: f32 = 1.0 / 64.0;
            let mut gains = vec![0.0f32; num * 3];
            let mut gain_min = [f32::MAX; 3];
            let mut gain_max = [f32::MIN; 3];

            for i in 0..num {
                let idx = i * 3;
                for c in 0..3 {
                    let gain = ((hdr_data[idx + c] * peak_ratio + offset)
                        / (data[idx + c] + offset))
                        .max(1e-10)
                        .log2();
                    gains[idx + c] = gain;
                    gain_min[c] = gain_min[c].min(gain);
                    gain_max[c] = gain_max[c].max(gain);
                }
            }

            let range = [
                (gain_max[0] - gain_min[0]).max(1e-6),
                (gain_max[1] - gain_min[1]).max(1e-6),
                (gain_max[2] - gain_min[2]).max(1e-6),
            ];
            let mut gain_rgb8 = vec![0u8; num * 3];
            for i in 0..num {
                let idx = i * 3;
                for c in 0..3 {
                    gain_rgb8[idx + c] =
                        ((gains[idx + c] - gain_min[c]) / range[c] * 255.0 + 0.5)
                            .clamp(0.0, 255.0) as u8;
                }
            }

            let exif_app1 = exif::build_orientation_app1(orientation);
            encode_uhdr::encode(
                &sdr_rgb8, width, height, quality,
                &gain_rgb8,
                gain_min, gain_max, offset,
                peak_ratio.log2(),
                &exif_app1,
            )
        }

        Format::Tiff => {
            // Physical rotation (tiff crate doesn't expose orientation tag easily)
            let orient = Orientation::parse(orientation);
            let (rotated, rw, rh) = rotation::apply_rotation(data, width, height, &orient);
            let num = (rw as usize) * (rh as usize);

            let mut rgb16 = vec![0u16; num * 3];
            for i in 0..num {
                let idx = i * 3;
                rgb16[idx]     = (rotated[idx].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
                rgb16[idx + 1] = (rotated[idx + 1].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
                rgb16[idx + 2] = (rotated[idx + 2].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
            }
            encode_tiff::encode(&rgb16, rw, rh)
        }

        Format::Avif => {
            // Physical rotation (avif-serialize doesn't support irot box)
            let orient = Orientation::parse(orientation);
            let (rotated, rw, rh) = rotation::apply_rotation(data, width, height, &orient);
            let num = (rw as usize) * (rh as usize);

            let mut rgb10 = vec![0u16; num * 3];
            for i in 0..num {
                let idx = i * 3;
                let r = rotated[idx].max(0.0).powf(1.0 / 1.2);
                let g = rotated[idx + 1].max(0.0).powf(1.0 / 1.2);
                let b = rotated[idx + 2].max(0.0).powf(1.0 / 1.2);
                rgb10[idx]     = (transfer::hlg_oetf(r) * 1023.0 + 0.5).clamp(0.0, 1023.0) as u16;
                rgb10[idx + 1] = (transfer::hlg_oetf(g) * 1023.0 + 0.5).clamp(0.0, 1023.0) as u16;
                rgb10[idx + 2] = (transfer::hlg_oetf(b) * 1023.0 + 0.5).clamp(0.0, 1023.0) as u16;
            }
            encode_avif::encode(&rgb10, rw, rh, quality)
        }
    }
}
