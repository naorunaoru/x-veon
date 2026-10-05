use crate::math;

/// sRGB OETF: linear [0,1] → sRGB non-linear [0,1]
#[inline]
pub fn srgb_oetf(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * math::powf(v, 1.0 / 2.4) - 0.055
    }
}

/// BT.2100 HLG OETF: linear scene light → non-linear HLG signal.
/// Input can exceed 1.0 (HDR content).
#[inline]
pub fn hlg_oetf(e: f32) -> f32 {
    let e = e.max(0.0);
    if e <= 1.0 / 12.0 {
        math::sqrtf(3.0 * e)
    } else {
        const A: f32 = 0.17883277;
        const B: f32 = 0.28466892;
        const C: f32 = 0.55991073;
        A * math::logf((12.0 * e - B).max(1e-10)) + C
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn hlg_reference_points() {
        assert!((hlg_oetf(1.0 / 12.0) - 0.5).abs() < 1e-6);
        assert!((hlg_oetf(1.0) - 1.0).abs() < 1e-4);
    }
    #[test] fn srgb_reference_points() {
        assert!((srgb_oetf(1.0) - 1.0).abs() < 1e-7);
        assert!((srgb_oetf(0.0031308) - 0.040449936).abs() < 1e-6);
    }
}
