pub type Mat3 = [[f32; 3]; 3];

pub const XYZ_TO_SRGB: Mat3 = [
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
];

pub const SRGB_TO_BT2020: Mat3 = [
    [0.6274039, 0.3292830, 0.0433131],
    [0.0690973, 0.9195404, 0.0113623],
    [0.0163914, 0.0880133, 0.8955953],
];

pub const IDENTITY: Mat3 = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
];

// ── P3-D65 working space (OpenDRT) ──────────────────────────────────────

/// XYZ → P3-D65 (ART CTL line 133)
pub const XYZ_TO_P3D65: Mat3 = [
    [ 2.49349691194, -0.931383617919, -0.402710784451],
    [-0.829488969562, 1.76266406032,   0.023624685842],
    [ 0.035845830244,-0.076172389268,  0.956884524008],
];

pub fn invert3x3(m: &Mat3) -> Mat3 {
    let [a, b, c] = m[0];
    let [d, e, f] = m[1];
    let [g, h, i] = m[2];

    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    let inv = 1.0 / det;

    [
        [(e * i - f * h) * inv, (c * h - b * i) * inv, (b * f - c * e) * inv],
        [(f * g - d * i) * inv, (a * i - c * g) * inv, (c * d - a * f) * inv],
        [(d * h - e * g) * inv, (b * g - a * h) * inv, (a * e - b * d) * inv],
    ]
}

pub fn mul3x3(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut r = [[0.0_f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    r
}

/// Build Camera→sRGB matrix using dcraw convention.
///
/// 1. sRGB→XYZ→Camera = forward matrix
/// 2. Row-normalize (sRGB white [1,1,1] → camera neutral [1,1,1])
/// 3. Invert to get Camera→sRGB
pub fn build_cam_to_srgb(xyz_to_cam: &Mat3) -> Mat3 {
    let srgb_to_xyz = invert3x3(&XYZ_TO_SRGB);
    let mut srgb_to_cam = mul3x3(xyz_to_cam, &srgb_to_xyz);

    for i in 0..3 {
        let sum: f32 = srgb_to_cam[i].iter().sum();
        for j in 0..3 {
            srgb_to_cam[i][j] /= sum;
        }
    }

    invert3x3(&srgb_to_cam)
}

/// Build Camera→BT.2020 = SRGB_TO_BT2020 * Camera→sRGB
pub fn build_cam_to_bt2020(xyz_to_cam: &Mat3) -> Mat3 {
    let cam_to_srgb = build_cam_to_srgb(xyz_to_cam);
    mul3x3(&SRGB_TO_BT2020, &cam_to_srgb)
}

/// Parse flat 9-element slice into Mat3.
pub fn mat3_from_slice(s: &[f32]) -> Mat3 {
    [
        [s[0], s[1], s[2]],
        [s[3], s[4], s[5]],
        [s[6], s[7], s[8]],
    ]
}

/// Build Camera→XYZ matrix from XYZ→Camera using dcraw row-normalization.
///
/// Uses the target→Camera forward matrix approach:
/// 1. Compose XYZ→Camera with target inverse to get target→Camera forward
/// 2. Row-normalize so white maps to [1,1,1]
/// 3. Invert to get Camera→target
fn build_cam_to_target(xyz_to_cam: &Mat3, xyz_to_target: &Mat3) -> Mat3 {
    let target_to_xyz = invert3x3(xyz_to_target);
    let mut target_to_cam = mul3x3(xyz_to_cam, &target_to_xyz);

    for i in 0..3 {
        let sum: f32 = target_to_cam[i].iter().sum();
        for j in 0..3 {
            target_to_cam[i][j] /= sum;
        }
    }

    invert3x3(&target_to_cam)
}

/// Build Camera→P3-D65 matrix for OpenDRT working space.
pub fn build_cam_to_p3d65(xyz_to_cam: &Mat3) -> Mat3 {
    build_cam_to_target(xyz_to_cam, &XYZ_TO_P3D65)
}
