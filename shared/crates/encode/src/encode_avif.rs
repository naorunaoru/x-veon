use rav1e::prelude::*;

/// Map quality 1-100 to rav1e quantizer (lower quantizer = higher quality).
fn quality_to_quantizer(quality: u8) -> usize {
    let q = (100 - quality.min(100)) as f32;
    (20.0 + q * 2.35).round() as usize
}

pub fn encode(rgb10: &[u16], width: u32, height: u32, quality: u8, options: crate::EncodeOptions) -> Result<Vec<u8>, String> {
    let w = width as usize;
    let h = height as usize;

    let enc = EncoderConfig {
        width: w,
        height: h,
        bit_depth: 10,
        chroma_sampling: ChromaSampling::Cs444,
        chroma_sample_position: ChromaSamplePosition::Unknown,
        pixel_range: PixelRange::Full,
        color_description: Some(ColorDescription {
            color_primaries: ColorPrimaries::BT2020,
            transfer_characteristics: TransferCharacteristics::HLG,
            matrix_coefficients: MatrixCoefficients::BT2020NCL,
        }),
        speed_settings: SpeedSettings::from_preset(10),
        quantizer: quality_to_quantizer(quality),
        min_quantizer: 0,
        low_latency: true,
        ..Default::default()
    };

    let cfg = Config::new().with_encoder_config(enc).with_threads(options.threads.max(1));
    let mut ctx: Context<u16> =
        cfg.new_context().map_err(|e| format!("rav1e context error: {e}"))?;

    // Full-range BT.2020 non-constant-luminance YCbCr, without subsampling.
    // Windows' HEIF decoder does not reliably handle identity-matrix HDR GBR.
    let mut frame = ctx.new_frame();
    let strides = [
        frame.planes[0].cfg.stride,
        frame.planes[1].cfg.stride,
        frame.planes[2].cfg.stride,
    ];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            let [luma, cb, cr] = rgb_to_ycbcr(rgb10[idx], rgb10[idx + 1], rgb10[idx + 2]);
            frame.planes[0].data_origin_mut()[y * strides[0] + x] = luma;
            frame.planes[1].data_origin_mut()[y * strides[1] + x] = cb;
            frame.planes[2].data_origin_mut()[y * strides[2] + x] = cr;
        }
    }

    ctx.send_frame(frame)
        .map_err(|e| format!("rav1e send_frame error: {e}"))?;
    ctx.flush();

    // Collect encoded AV1 packets
    let mut av1_data = Vec::new();
    loop {
        match ctx.receive_packet() {
            Ok(pkt) => av1_data.extend_from_slice(&pkt.data),
            Err(EncoderStatus::LimitReached) => break,
            Err(EncoderStatus::Encoded) | Err(EncoderStatus::NeedMoreData) => {}
            Err(e) => return Err(format!("rav1e encode error: {e:?}")),
        }
    }

    if av1_data.is_empty() {
        return Err("rav1e produced no output".into());
    }

    // Match the AV1 payload in the container too, so readers do not assume sRGB/BT.601.
    let avif = avif_serialize::Aviffy::new()
        .premultiplied_alpha(false)
        .set_color_primaries(avif_serialize::constants::ColorPrimaries::Bt2020)
        .set_transfer_characteristics(avif_serialize::constants::TransferCharacteristics::Hlg)
        .set_matrix_coefficients(avif_serialize::constants::MatrixCoefficients::Bt2020Ncl)
        .set_full_color_range(true)
        .to_vec(&av1_data, None, width, height, 10);

    Ok(avif)
}

fn rgb_to_ycbcr(r: u16, g: u16, b: u16) -> [u16; 3] {
    // Kr=0.2627, Kg=0.6780, Kb=0.0593. Integer arithmetic keeps native and WASM
    // rounding identical; 10-bit full-range chroma is centered at 512.
    let (r, g, b) = (i32::from(r), i32::from(g), i32::from(b));
    let y = 2627 * r + 6780 * g + 593 * b;
    let round = |n: i32, d: i32| if n < 0 { -((-n + d / 2) / d) } else { (n + d / 2) / d };
    [round(y, 10000), 512 + round(10000 * b - y, 18814), 512 + round(10000 * r - y, 14746)]
        .map(|v| v.clamp(0, 1023) as u16)
}

#[cfg(test)]
mod tests {
    use super::rgb_to_ycbcr;
    #[test]
    fn bt2020_neutrals_and_primaries() {
        for v in [0, 1, 256, 512, 1023] { assert_eq!(rgb_to_ycbcr(v, v, v), [v, 512, 512]); }
        assert_eq!(rgb_to_ycbcr(1023, 0, 0), [269, 369, 1023]);
        assert_eq!(rgb_to_ycbcr(0, 1023, 0), [694, 143, 42]);
        assert_eq!(rgb_to_ycbcr(0, 0, 1023), [61, 1023, 471]);
    }
}
