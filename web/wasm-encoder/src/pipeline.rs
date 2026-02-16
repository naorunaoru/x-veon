use crate::color::{self, Mat3};
use crate::encode_avif;
use crate::encode_tiff;
use crate::encode_uhdr;
use crate::opendrt::{self, OpenDrtConfig, TonescaleParams, LookPreset};
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

pub fn encode(
    hwc: &[f32],
    width: u32,
    height: u32,
    xyz_to_cam: &Mat3,
    orientation: &str,
    format: Format,
    quality: u8,
    look_preset: LookPreset,
) -> Result<Vec<u8>, String> {
    let cc_applied = xyz_to_cam.iter().all(|row| row.iter().all(|v| v.abs() < 1e-10));
    let cam_to_p3 = if cc_applied {
        // Data already in sRGB — approximate: use sRGB→XYZ→P3 (no camera matrix)
        let srgb_to_xyz = color::invert3x3(&color::XYZ_TO_SRGB);
        color::mul3x3(&color::XYZ_TO_P3D65, &srgb_to_xyz)
    } else {
        color::build_cam_to_p3d65(xyz_to_cam)
    };

    // EXIF rotation first — apply to raw input, then tone-map the rotated buffer.
    // (Rotation is just a transpose/flip, order doesn't matter for pixel-independent ops,
    //  but doing it first keeps the rest format-specific code simpler.)
    let orient = Orientation::parse(orientation);
    let (rotated, rw, rh) = rotation::apply_rotation(hwc, width, height, &orient);
    let num = (rw as usize) * (rh as usize);

    match format {
        Format::JpegHdr => {
            // Dual render: SDR (Rec.709) + HDR (Rec.2020) from same scene data.
            let cfg_sdr = OpenDrtConfig::from_preset(look_preset, false);
            let cfg_hdr = OpenDrtConfig::from_preset(look_preset, true);
            let ts_sdr = TonescaleParams::new(&cfg_sdr);
            let ts_hdr = TonescaleParams::new(&cfg_hdr);

            let mut sdr_lin = vec![0.0f32; num * 3];
            let mut sdr_rgb8 = vec![0u8; num * 3];
            let mut hdr_lin = vec![0.0f32; num * 3];

            for i in 0..num {
                let idx = i * 3;
                let px_in = [rotated[idx], rotated[idx + 1], rotated[idx + 2]];

                let sdr = opendrt::process_pixel(px_in, &ts_sdr, &cfg_sdr, &cam_to_p3);
                sdr_lin[idx]     = sdr[0];
                sdr_lin[idx + 1] = sdr[1];
                sdr_lin[idx + 2] = sdr[2];
                sdr_rgb8[idx]     = (transfer::srgb_oetf(sdr[0]) * 255.0 + 0.5) as u8;
                sdr_rgb8[idx + 1] = (transfer::srgb_oetf(sdr[1]) * 255.0 + 0.5) as u8;
                sdr_rgb8[idx + 2] = (transfer::srgb_oetf(sdr[2]) * 255.0 + 0.5) as u8;

                let hdr = opendrt::process_pixel(px_in, &ts_hdr, &cfg_hdr, &cam_to_p3);
                hdr_lin[idx]     = hdr[0];
                hdr_lin[idx + 1] = hdr[1];
                hdr_lin[idx + 2] = hdr[2];
            }

            // Per-channel gain map: log2(hdr_c / sdr_c) independently per RGB channel.
            // HDR output is normalized to [0,1] relative to its peak (1000 nit),
            // SDR to its peak (100 nit). Scale HDR to SDR-relative nits for correct ratio.
            let peak_ratio = cfg_hdr.peak_luminance / cfg_sdr.peak_luminance;
            let offset: f32 = 1.0 / 64.0;
            let mut gains = vec![0.0f32; num * 3];
            let mut gain_min = [f32::MAX; 3];
            let mut gain_max = [f32::MIN; 3];

            for i in 0..num {
                let idx = i * 3;
                for c in 0..3 {
                    let gain = ((hdr_lin[idx + c] * peak_ratio + offset) / (sdr_lin[idx + c] + offset)).max(1e-10).log2();
                    gains[idx + c] = gain;
                    gain_min[c] = gain_min[c].min(gain);
                    gain_max[c] = gain_max[c].max(gain);
                }
            }

            // Normalize to [0, 255] per channel
            let range = [
                (gain_max[0] - gain_min[0]).max(1e-6),
                (gain_max[1] - gain_min[1]).max(1e-6),
                (gain_max[2] - gain_min[2]).max(1e-6),
            ];
            let mut gain_rgb8 = vec![0u8; num * 3];
            for i in 0..num {
                let idx = i * 3;
                for c in 0..3 {
                    gain_rgb8[idx + c] = ((gains[idx + c] - gain_min[c]) / range[c] * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
                }
            }

            encode_uhdr::encode(
                &sdr_rgb8, rw, rh, quality,
                &gain_rgb8,
                gain_min, gain_max, offset,
                peak_ratio.log2(),
            )
        }

        Format::Tiff => {
            // Linear SDR in Rec.709, 16-bit
            let cfg = OpenDrtConfig::from_preset(look_preset, false);
            let ts = TonescaleParams::new(&cfg);
            let mut rgb16 = vec![0u16; num * 3];
            for i in 0..num {
                let idx = i * 3;
                let px = opendrt::process_pixel(
                    [rotated[idx], rotated[idx + 1], rotated[idx + 2]],
                    &ts, &cfg, &cam_to_p3,
                );
                rgb16[idx]     = (px[0].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
                rgb16[idx + 1] = (px[1].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
                rgb16[idx + 2] = (px[2].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
            }
            encode_tiff::encode(&rgb16, rw, rh)
        }

        Format::Avif => {
            // HDR: OpenDRT → display-linear Rec.2020 → inverse OOTF → HLG OETF → 10-bit
            let cfg = OpenDrtConfig::from_preset(look_preset, true);
            let ts = TonescaleParams::new(&cfg);
            // BT.2100 OOTF gamma for reference 1000-nit display:
            // γ = 1.2 + 0.42 * log10(Lw/1000). At Lw=1000: γ=1.2.
            // Inverse OOTF converts display-linear → scene-linear so the
            // display's own OOTF recovers the intended luminance.
            let mut rgb10 = vec![0u16; num * 3];
            for i in 0..num {
                let idx = i * 3;
                let px = opendrt::process_pixel(
                    [rotated[idx], rotated[idx + 1], rotated[idx + 2]],
                    &ts, &cfg, &cam_to_p3,
                );
                // OpenDRT output is display-linear [0,1] where 1.0 = peak luminance.
                // Apply approximate inverse OOTF: scene = display^(1/γ).
                // BT.2100 OOTF is luminance-weighted: Fd = Lw * Ys^(γ-1) * Es
                // Inverse: Es = Fd / (Lw * Ys^(γ-1))
                // Since OpenDRT output is normalized (Fd/Lw), we compute:
                //   Ys = 0.2627*R + 0.6780*G + 0.0593*B (BT.2020 luminance)
                //   Es = (Fd/Lw) / Ys^(γ-1) * Ys = (Fd/Lw) * Ys^(1-(γ-1)) ... complicated
                // Simplified per-channel approximation: scene ≈ display^(1/γ)
                let r = px[0].max(0.0).powf(1.0 / 1.2);
                let g = px[1].max(0.0).powf(1.0 / 1.2);
                let b = px[2].max(0.0).powf(1.0 / 1.2);
                rgb10[idx]     = (transfer::hlg_oetf(r) * 1023.0 + 0.5).clamp(0.0, 1023.0) as u16;
                rgb10[idx + 1] = (transfer::hlg_oetf(g) * 1023.0 + 0.5).clamp(0.0, 1023.0) as u16;
                rgb10[idx + 2] = (transfer::hlg_oetf(b) * 1023.0 + 0.5).clamp(0.0, 1023.0) as u16;
            }
            encode_avif::encode(&rgb10, rw, rh, quality)
        }
    }
}
