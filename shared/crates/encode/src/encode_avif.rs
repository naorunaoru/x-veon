use rav1e::prelude::*;

/// Map quality 1-100 to rav1e quantizer (lower quantizer = higher quality).
fn quality_to_quantizer(quality: u8) -> usize {
    let q = (100 - quality.min(100)) as f32;
    (20.0 + q * 2.35).round() as usize
}

pub fn encode(rgb10: &[u16], width: u32, height: u32, quality: u8) -> Result<Vec<u8>, String> {
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
            matrix_coefficients: MatrixCoefficients::Identity,
        }),
        speed_settings: SpeedSettings::from_preset(10),
        quantizer: quality_to_quantizer(quality),
        min_quantizer: 0,
        low_latency: true,
        ..Default::default()
    };

    let cfg = Config::new().with_encoder_config(enc).with_threads(1);
    let mut ctx: Context<u16> =
        cfg.new_context().map_err(|e| format!("rav1e context error: {e}"))?;

    // Build frame — Identity matrix means planes are G, B, R (IEC 61966-2-1 convention).
    // For Identity MC in AV1, plane 0 = Green, plane 1 = Blue, plane 2 = Red.
    let mut frame = ctx.new_frame();
    let strides = [
        frame.planes[0].cfg.stride,
        frame.planes[1].cfg.stride,
        frame.planes[2].cfg.stride,
    ];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            let r = rgb10[idx];
            let g = rgb10[idx + 1];
            let b = rgb10[idx + 2];
            frame.planes[0].data_origin_mut()[y * strides[0] + x] = g;
            frame.planes[1].data_origin_mut()[y * strides[1] + x] = b;
            frame.planes[2].data_origin_mut()[y * strides[2] + x] = r;
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

    // Wrap in AVIF container with CICP tags
    let avif = avif_serialize::Aviffy::new()
        .premultiplied_alpha(false)
        .to_vec(&av1_data, None, width, height, 10);

    Ok(avif)
}
