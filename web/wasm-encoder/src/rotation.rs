pub enum Orientation {
    Normal,
    Rotate90,
    Rotate180,
    Rotate270,
}

impl Orientation {
    pub fn parse(s: &str) -> Self {
        match s {
            "Rotate90" => Orientation::Rotate90,
            "Rotate180" => Orientation::Rotate180,
            "Rotate270" => Orientation::Rotate270,
            _ => Orientation::Normal,
        }
    }
}

/// Apply physical EXIF rotation to HWC float buffer.
/// Returns (rotated_data, new_width, new_height).
pub fn apply_rotation(
    hwc: &[f32],
    width: u32,
    height: u32,
    orientation: &Orientation,
) -> (Vec<f32>, u32, u32) {
    let w = width as usize;
    let h = height as usize;
    let n = w * h;

    match orientation {
        Orientation::Normal => (hwc.to_vec(), width, height),

        Orientation::Rotate180 => {
            let mut out = vec![0.0_f32; n * 3];
            for i in 0..n {
                let si = (n - 1 - i) * 3;
                let di = i * 3;
                out[di] = hwc[si];
                out[di + 1] = hwc[si + 1];
                out[di + 2] = hwc[si + 2];
            }
            (out, width, height)
        }

        Orientation::Rotate90 => {
            // 90° CW: new dimensions are (height, width) → newW=h, newH=w
            let new_w = h;
            let new_h = w;
            let mut out = vec![0.0_f32; n * 3];
            for y in 0..h {
                for x in 0..w {
                    let si = (y * w + x) * 3;
                    let dx = h - 1 - y;
                    let dy = x;
                    let di = (dy * new_w + dx) * 3;
                    out[di] = hwc[si];
                    out[di + 1] = hwc[si + 1];
                    out[di + 2] = hwc[si + 2];
                }
            }
            (out, new_w as u32, new_h as u32)
        }

        Orientation::Rotate270 => {
            // 90° CCW: newW=h, newH=w
            let new_w = h;
            let new_h = w;
            let mut out = vec![0.0_f32; n * 3];
            for y in 0..h {
                for x in 0..w {
                    let si = (y * w + x) * 3;
                    let dx = y;
                    let dy = w - 1 - x;
                    let di = (dy * new_w + dx) * 3;
                    out[di] = hwc[si];
                    out[di + 1] = hwc[si + 1];
                    out[di + 2] = hwc[si + 2];
                }
            }
            (out, new_w as u32, new_h as u32)
        }
    }
}
