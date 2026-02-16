use crate::color::{self, Mat3};
use crate::encode_avif;
use crate::encode_jpeg;
use crate::encode_tiff;
use crate::encode_uhdr;
use crate::opendrt::{self, OpenDrtConfig, TonescaleParams, LookPreset};
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

#[derive(Clone, Copy, PartialEq)]
pub enum ToneMap {
    Legacy,
    OpenDrt,
}

impl ToneMap {
    pub fn parse(s: &str) -> Self {
        match s {
            "opendrt" => ToneMap::OpenDrt,
            _ => ToneMap::Legacy,
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
    tone_map: ToneMap,
    look_preset: LookPreset,
) -> Result<Vec<u8>, String> {
    if tone_map == ToneMap::OpenDrt {
        return encode_opendrt(hwc, width, height, xyz_to_cam, orientation, format, quality, look_preset);
    }

    // ── Legacy path ──────────────────────────────────────────────────────

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
            // Ultra HDR JPEG: SDR base + gain map for highlight recovery.
            // HLG OETF for peak normalization (same as AVIF) to preserve mid-tone brightness.
            // TODO: bake HLG OOTF (system gamma 1.2) to match AVIF display contrast
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

            // SDR base: HLG-normalized → linear (tone-mapped) → sRGB 8-bit
            let mut sdr_lin = vec![0.0f32; num * 3];
            let mut sdr_rgb8 = vec![0u8; num * 3];
            for i in 0..num * 3 {
                let lin = transfer::hlg_oetf_inv(hlg_buf[i] * scale);
                sdr_lin[i] = lin;
                sdr_rgb8[i] = (transfer::srgb_oetf(lin) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            }

            // Gain map: per-pixel luminance log2 ratio.
            let offset: f32 = 1.0 / 64.0;
            let mut gains = vec![0.0f32; num];
            let mut gain_min: f32 = f32::MAX;
            let mut gain_max: f32 = f32::MIN;

            for i in 0..num {
                let idx = i * 3;
                let y_hdr = 0.2126 * rotated[idx] + 0.7152 * rotated[idx + 1] + 0.0722 * rotated[idx + 2];
                let y_sdr = 0.2126 * sdr_lin[idx]
                    + 0.7152 * sdr_lin[idx + 1]
                    + 0.0722 * sdr_lin[idx + 2];
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
                gain_max, // legacy: content peak IS the capacity
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

// ── OpenDRT path ─────────────────────────────────────────────────────────

fn encode_opendrt(
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
    let orient = rotation::Orientation::parse(orientation);
    let (rotated, rw, rh) = rotation::apply_rotation(hwc, width, height, &orient);
    let num = (rw as usize) * (rh as usize);

    match format {
        Format::Jpeg => {
            let cfg = OpenDrtConfig::from_preset(look_preset, false);
            let ts = TonescaleParams::new(&cfg);
            let mut rgb8 = vec![0u8; num * 3];
            for i in 0..num {
                let idx = i * 3;
                let px = opendrt::process_pixel(
                    [rotated[idx], rotated[idx + 1], rotated[idx + 2]],
                    &ts, &cfg, &cam_to_p3,
                );
                rgb8[idx]     = (transfer::srgb_oetf(px[0]) * 255.0 + 0.5) as u8;
                rgb8[idx + 1] = (transfer::srgb_oetf(px[1]) * 255.0 + 0.5) as u8;
                rgb8[idx + 2] = (transfer::srgb_oetf(px[2]) * 255.0 + 0.5) as u8;
            }
            encode_jpeg::encode(&rgb8, rw, rh, quality)
        }

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

            // Gain map: per-pixel luminance log2 ratio.
            // HDR output is normalized to [0,1] relative to its peak (1000 nit),
            // SDR to its peak (100 nit). Scale HDR to SDR-relative nits for correct ratio.
            let peak_ratio = cfg_hdr.peak_luminance / cfg_sdr.peak_luminance;
            let offset: f32 = 1.0 / 64.0;
            let mut gains = vec![0.0f32; num];
            let mut gain_min: f32 = f32::MAX;
            let mut gain_max: f32 = f32::MIN;

            for i in 0..num {
                let idx = i * 3;
                let y_hdr = 0.2627 * hdr_lin[idx] + 0.6780 * hdr_lin[idx + 1] + 0.0593 * hdr_lin[idx + 2];
                let y_sdr = 0.2126 * sdr_lin[idx] + 0.7152 * sdr_lin[idx + 1] + 0.0722 * sdr_lin[idx + 2];
                let gain = ((y_hdr * peak_ratio + offset) / (y_sdr + offset)).max(1e-10).log2();
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
