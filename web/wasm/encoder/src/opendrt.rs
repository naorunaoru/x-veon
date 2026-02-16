//! OpenDRT v1.0.0 — display rendering transform with look presets.
//!
//! Ported from opendrt_art.ctl (ART CTL by agriggio, based on Jed Smith's OpenDRT).
//! License: GPLv3
//!
//! Input:  scene-linear P3-D65 RGB
//! Output: display-linear RGB in target gamut (Rec.709 or Rec.2020), before EOTF

use crate::color::Mat3;

const SQRT3: f32 = 1.7320508;
const PI: f32 = core::f32::consts::PI;

// ── P3-D65 → display gamut matrices (ART CTL lines 142-143) ─────────────

const P3D65_TO_REC709: Mat3 = [
    [ 1.224940181, -0.2249402404,  0.0],
    [-0.04205697775, 1.042057037, -1.4901e-08],
    [-0.01963755488,-0.07863604277, 1.098273635],
];

const P3D65_TO_REC2020: Mat3 = [
    [0.7538330344, 0.1985973691, 0.04756959659],
    [0.04574384897, 0.9417772198, 0.01247893122],
    [-0.001210340355, 0.0176017173, 0.9836086231],
];

// ── Display gamut enum ───────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
pub enum DisplayGamut {
    Rec709,
    Rec2020,
}

// ── Look preset enum ────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
pub enum LookPreset {
    Base,
    Default,
}

// ── Config (ART CTL lines 878-1077) ──────────────────────────────────────

pub struct OpenDrtConfig {
    // Tonescale
    pub tn_lg: f32,
    pub tn_con: f32,
    pub tn_sh: f32,
    pub tn_toe: f32,
    pub tn_off: f32,
    // Contrast Low (CTL 553-585)
    pub tn_lcon_enable: bool,
    pub tn_lcon: f32,
    pub tn_lcon_w: f32,
    pub tn_lcon_pc: f32,
    // Contrast High (CTL 599-603)
    pub tn_hcon_enable: bool,
    pub tn_hcon: f32,
    pub tn_hcon_pv: f32,
    pub tn_hcon_st: f32,
    // Rendering space
    pub rs_sa: f32,
    pub rs_rw: f32,
    pub rs_bw: f32,
    // Purity compression
    pub pt_r: f32,
    pub pt_g: f32,
    pub pt_b: f32,
    pub pt_rng_low: f32,
    pub pt_rng_high: f32,
    pub ptl_enable: bool,
    // Mid purity (CTL 664-679)
    pub ptm_enable: bool,
    pub ptm_low: f32,
    pub ptm_low_st: f32,
    pub ptm_high: f32,
    pub ptm_high_st: f32,
    // Brilliance (CTL 646-660)
    pub brl_enable: bool,
    pub brl_r: f32,
    pub brl_g: f32,
    pub brl_b: f32,
    pub brl_c: f32,
    pub brl_m: f32,
    pub brl_y: f32,
    pub brl_rng: f32,
    // Display
    pub peak_luminance: f32,
    pub grey_boost: f32,
    pub pt_hdr: f32,
    pub display_gamut: DisplayGamut,
}

impl OpenDrtConfig {
    /// Base preset: minimal, all look modules disabled. (ART CTL lines 1028-1077)
    pub fn base_sdr() -> Self {
        Self {
            tn_lg: 18.0,
            tn_con: 1.4,
            tn_sh: 0.5,
            tn_toe: 0.003,
            tn_off: 0.0,
            tn_lcon_enable: false,
            tn_lcon: 0.0,
            tn_lcon_w: 0.5,
            tn_lcon_pc: 1.0,
            tn_hcon_enable: false,
            tn_hcon: 0.0,
            tn_hcon_pv: 1.0,
            tn_hcon_st: 4.0,
            rs_sa: 0.35,
            rs_rw: 0.25,
            rs_bw: 0.5,
            pt_r: 1.0,
            pt_g: 2.0,
            pt_b: 2.5,
            pt_rng_low: 0.25,
            pt_rng_high: 0.25,
            ptl_enable: true,
            ptm_enable: false,
            ptm_low: 0.0,
            ptm_low_st: 0.5,
            ptm_high: 0.0,
            ptm_high_st: 0.3,
            brl_enable: false,
            brl_r: 0.0,
            brl_g: 0.0,
            brl_b: 0.0,
            brl_c: 0.0,
            brl_m: 0.0,
            brl_y: 0.0,
            brl_rng: 0.5,
            peak_luminance: 100.0,
            grey_boost: 0.13,
            pt_hdr: 0.5,
            display_gamut: DisplayGamut::Rec709,
        }
    }

    pub fn base_hdr() -> Self {
        Self {
            peak_luminance: 1000.0,
            display_gamut: DisplayGamut::Rec2020,
            ..Self::base_sdr()
        }
    }

    /// Default preset: contrast low, brilliance, mid purity enabled. (ART CTL lines 878-927)
    pub fn default_sdr() -> Self {
        Self {
            tn_off: 0.005,
            tn_lcon_enable: true,
            tn_lcon: 1.0,
            tn_lcon_w: 0.5,
            tn_lcon_pc: 1.0,
            rs_bw: 0.55,
            pt_r: 0.5,
            pt_g: 2.0,
            pt_b: 2.0,
            pt_rng_low: 0.2,
            pt_rng_high: 0.8,
            ptm_enable: true,
            ptm_low: 0.2,
            ptm_low_st: 0.5,
            ptm_high: -0.8,
            ptm_high_st: 0.3,
            brl_enable: true,
            brl_r: -0.5,
            brl_g: -0.4,
            brl_b: -0.2,
            brl_c: 0.0,
            brl_m: 0.0,
            brl_y: 0.0,
            brl_rng: 0.66,
            ..Self::base_sdr()
        }
    }

    pub fn default_hdr() -> Self {
        Self {
            peak_luminance: 1000.0,
            display_gamut: DisplayGamut::Rec2020,
            ..Self::default_sdr()
        }
    }

    pub fn from_preset(preset: LookPreset, hdr: bool) -> Self {
        match (preset, hdr) {
            (LookPreset::Base, false) => Self::base_sdr(),
            (LookPreset::Base, true) => Self::base_hdr(),
            (LookPreset::Default, false) => Self::default_sdr(),
            (LookPreset::Default, true) => Self::default_hdr(),
        }
    }
}

// ── Pre-computed tonescale constants (pixel-independent) ─────────────────

pub struct TonescaleParams {
    ts_s: f32,
    ts_s1: f32,
    ts_m2: f32,
    ts_dsc: f32,
    ts_x0: f32,
}

impl TonescaleParams {
    /// CTL lines 516-531: constraint solver.
    pub fn new(cfg: &OpenDrtConfig) -> Self {
        let ts_x1 = f32_exp2(6.0 * cfg.tn_sh + 4.0);
        let ts_y1 = cfg.peak_luminance / 100.0;
        let ts_x0 = 0.18 + cfg.tn_off;
        let ts_y0 = (cfg.tn_lg / 100.0) * (1.0 + cfg.grey_boost * f32_log2(ts_y1.max(1e-10)));
        let ts_s0 = compress_toe_quadratic_inv(ts_y0, cfg.tn_toe);
        let ts_s10 = ts_x0 * (spowf(ts_s0, -1.0 / cfg.tn_con) - 1.0);
        let ts_m1 = ts_y1 / spowf(ts_x1 / (ts_x1 + ts_s10), cfg.tn_con);
        let ts_m2 = compress_toe_quadratic_inv(ts_m1, cfg.tn_toe);
        let ts_s = ts_x0 * (spowf(ts_s0 / ts_m2, -1.0 / cfg.tn_con) - 1.0);
        let ts_dsc = 100.0 / cfg.peak_luminance;

        // HDR purity blend
        let pt_cmp_lf = cfg.pt_hdr * ((cfg.peak_luminance - 100.0) / 900.0).min(1.0);
        let s_lp100 = ts_x0 * (spowf(cfg.tn_lg / 100.0, -1.0 / cfg.tn_con) - 1.0);
        let ts_s1 = ts_s * pt_cmp_lf + s_lp100 * (1.0 - pt_cmp_lf);

        Self { ts_s, ts_s1, ts_m2, ts_dsc, ts_x0 }
    }
}

// ── Math helpers (CTL lines 157-423) ─────────────────────────────────────

/// Safe power: returns x unchanged if x <= 0.
#[inline(always)]
fn spowf(x: f32, p: f32) -> f32 {
    if x <= 0.0 { x } else { x.powf(p) }
}

/// Safe division.
#[inline(always)]
fn sdivf(a: f32, b: f32) -> f32 {
    if b == 0.0 { 0.0 } else { a / b }
}

/// Hyperbolic power compression: x/(x+s)^p. CTL line 338-342.
#[inline(always)]
fn compress_hp(x: f32, s: f32, p: f32) -> f32 {
    spowf(x / (x + s), p)
}

/// Quadratic toe forward: x²/(x+toe). CTL line 348-349.
#[inline(always)]
fn compress_toe_quadratic(x: f32, toe: f32) -> f32 {
    if toe == 0.0 { return x; }
    spowf(x, 2.0) / (x + toe)
}

/// Quadratic toe inverse: (x+sqrt(x*(4*toe+x)))/2. CTL line 350-352.
#[inline(always)]
fn compress_toe_quadratic_inv(x: f32, toe: f32) -> f32 {
    if toe == 0.0 { return x; }
    (x + (x * (4.0 * toe + x)).sqrt()) / 2.0
}

/// Cubic toe. CTL line 355-368.
/// Forward: x*(x²+m*w)/(x²+w). Inverse: Cardano's formula.
#[inline(always)]
fn compress_toe_cubic(x: f32, m: f32, w: f32, inv: bool) -> f32 {
    if m == 1.0 { return x; }
    let x2 = x * x;
    if !inv {
        x * (x2 + m * w) / (x2 + w)
    } else {
        let p0 = x2 - 3.0 * m * w;
        let p1 = 2.0 * x2 + 27.0 * w - 9.0 * m * w;
        let p2 = ((x2 * p1 * p1 - 4.0 * p0 * p0 * p0).max(0.0).sqrt() / 2.0 + x * p1 / 2.0).powf(1.0 / 3.0);
        p0 / (3.0 * p2) + p2 / 3.0 + x / 3.0
    }
}

/// High contrast with linear extension. CTL line 382-399.
#[inline(always)]
fn contrast_high(x: f32, p: f32, pv: f32, pv_lx: f32) -> f32 {
    let x0 = 0.18 * (2.0_f32).powf(pv);
    if x < x0 || p == 1.0 { return x; }
    let o = x0 - x0 / p;
    let s0 = x0.powf(1.0 - p) / p;
    let x1 = x0 * (2.0_f32).powf(pv_lx);
    let k1 = p * s0 * x1.powf(p) / x1;
    let y1 = s0 * x1.powf(p) + o;
    if x > x1 { k1 * (x - x1) + y1 } else { s0 * x.powf(p) + o }
}

/// Complement power: 1 - (1-x)^(1/p). CTL line 370-373.
#[inline(always)]
fn complement_power(x: f32, p: f32) -> f32 {
    1.0 - spowf(1.0 - x, 1.0 / p)
}

/// Cubic sigmoid: 1 + s*(1 - 3x² + 2x³). CTL line 375-380.
#[inline(always)]
fn sigmoid_cubic(x: f32, s: f32) -> f32 {
    if x < 0.0 || x > 1.0 { return 1.0; }
    1.0 + s * (1.0 - 3.0 * x * x + 2.0 * x * x * x)
}

/// Gaussian window. CTL line 411-416.
#[inline(always)]
fn gauss_window(x: f32, w: f32) -> f32 {
    let y = x / w;
    (-y * y).exp()
}

/// Offset hue maintaining 0-2π range. CTL line 419-423.
#[inline(always)]
fn hue_offset(h: f32, o: f32) -> f32 {
    ((h - o + PI) % (2.0 * PI)) - PI
}

/// Softplus smooth clamp. CTL line 401-408.
#[inline(always)]
fn softplus(x: f32, s: f32, x0: f32, y0: f32) -> f32 {
    if x > 10.0 * s + y0 || s < 1e-3 {
        return x;
    }
    let mut m: f32 = 1.0;
    if y0.abs() > 1e-6 {
        m = (y0 / s).exp();
    }
    m -= (x0 / s).exp();
    s * (m + (x / s).exp()).max(0.0).ln()
}

/// log2 for f32.
#[inline(always)]
fn f32_log2(x: f32) -> f32 {
    x.ln() / core::f32::consts::LN_2
}

/// 2^x for f32.
#[inline(always)]
fn f32_exp2(x: f32) -> f32 {
    (x * core::f32::consts::LN_2).exp()
}

/// Apply 3x3 matrix to RGB triplet.
#[inline(always)]
fn mat3_apply(m: &Mat3, rgb: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * rgb[0] + m[0][1] * rgb[1] + m[0][2] * rgb[2],
        m[1][0] * rgb[0] + m[1][1] * rgb[1] + m[1][2] * rgb[2],
        m[2][0] * rgb[0] + m[2][1] * rgb[1] + m[2][2] * rgb[2],
    ]
}

// ── Core per-pixel transform ─────────────────────────────────────────────

/// Process a single pixel through OpenDRT.
///
/// Input: scene-linear camera RGB (WB'd).
/// Output: display-linear RGB in target gamut, clamped [0,1], before EOTF.
#[inline(always)]
pub fn process_pixel(
    rgb_in: [f32; 3],
    ts: &TonescaleParams,
    cfg: &OpenDrtConfig,
    cam_to_p3: &Mat3,
) -> [f32; 3] {
    // ── Stage 1: Camera → P3-D65 ──
    let mut r;
    let mut g;
    let mut b;
    {
        let p = mat3_apply(cam_to_p3, rgb_in);
        r = p[0];
        g = p[1];
        b = p[2];
    }

    // ── Stage 2: Rendering space saturation + offset (CTL 539-547) ──
    let rs_gw = 1.0 - cfg.rs_rw - cfg.rs_bw;
    let sat_l = r * cfg.rs_rw + g * rs_gw + b * cfg.rs_bw;
    r = sat_l * cfg.rs_sa + r * (1.0 - cfg.rs_sa);
    g = sat_l * cfg.rs_sa + g * (1.0 - cfg.rs_sa);
    b = sat_l * cfg.rs_sa + b * (1.0 - cfg.rs_sa);

    // Offset
    r += cfg.tn_off;
    g += cfg.tn_off;
    b += cfg.tn_off;

    // ── Contrast Low (CTL 553-585) ──
    if cfg.tn_lcon_enable {
        let mcon_m = f32_exp2(-cfg.tn_lcon);
        let mcon_w_half = cfg.tn_lcon_w / 4.0;
        let mcon_w = mcon_w_half * mcon_w_half;

        // Normalize for ts_x0 intersection constraint (CTL 559)
        let mcon_cnst_sc = compress_toe_cubic(ts.ts_x0, mcon_m, mcon_w, true) / ts.ts_x0;
        r *= mcon_cnst_sc;
        g *= mcon_cnst_sc;
        b *= mcon_cnst_sc;

        // Ratio-preserving midtone contrast scale (CTL 563-564)
        let mcon_nm = (r.max(0.0) * r.max(0.0) + g.max(0.0) * g.max(0.0) + b.max(0.0) * b.max(0.0)).sqrt() / SQRT3;
        let mcon_sc = (mcon_nm * mcon_nm + mcon_m * mcon_w) / (mcon_nm * mcon_nm + mcon_w);

        if cfg.tn_lcon_pc > 0.0 {
            // Per-channel midtone contrast (CTL 570-573)
            let mcon_r = compress_toe_cubic(r, mcon_m, mcon_w, false);
            let mcon_g = compress_toe_cubic(g, mcon_m, mcon_w, false);
            let mcon_b = compress_toe_cubic(b, mcon_m, mcon_w, false);

            // Blend ratio-preserving vs per-channel based on distance from achromatic (CTL 576-580)
            let mcon_mx = r.max(g).max(b);
            let mcon_mn = r.min(g).min(b);
            let mut mcon_ch = (1.0 - sdivf(mcon_mn, mcon_mx)).clamp(0.0, 1.0);
            mcon_ch = mcon_ch.powf(4.0 * cfg.tn_lcon_pc);
            r = r * mcon_sc * mcon_ch + mcon_r * (1.0 - mcon_ch);
            g = g * mcon_sc * mcon_ch + mcon_g * (1.0 - mcon_ch);
            b = b * mcon_sc * mcon_ch + mcon_b * (1.0 - mcon_ch);
        } else {
            r *= mcon_sc;
            g *= mcon_sc;
            b *= mcon_sc;
        }
    }

    // ── Tonescale norm (CTL 588-594) ──
    let rc = r.max(0.0);
    let gc = g.max(0.0);
    let bc = b.max(0.0);
    let mut tsn = (rc * rc + gc * gc + bc * bc).sqrt() / SQRT3;

    // Purity compression norm (CTL 591)
    let mut ts_pt = (r * r * cfg.pt_r + g * g * cfg.pt_g + b * b * cfg.pt_b)
        .max(0.0)
        .sqrt();

    // RGB ratios (CTL 594): clamp to -2, divide by tsn
    if tsn > 0.0 {
        let inv = 1.0 / tsn;
        r = r.max(-2.0) * inv;
        g = g.max(-2.0) * inv;
        b = b.max(-2.0) * inv;
    } else {
        r = 0.0;
        g = 0.0;
        b = 0.0;
    }

    // ── Contrast High (CTL 599-603) ──
    if cfg.tn_hcon_enable {
        let hcon_p = f32_exp2(cfg.tn_hcon);
        tsn = contrast_high(tsn, hcon_p, cfg.tn_hcon_pv, cfg.tn_hcon_st);
        ts_pt = contrast_high(ts_pt, hcon_p, cfg.tn_hcon_pv, cfg.tn_hcon_st);
    }

    // ── Apply tonescale (CTL 605-607) ──
    tsn = compress_hp(tsn, ts.ts_s, cfg.tn_con);
    ts_pt = compress_hp(ts_pt, ts.ts_s1, cfg.tn_con);

    // ── Opponent space / achromatic distance (CTL 610-616) ──
    let opp_cy = r - b;
    let opp_gm = g - (r + b) / 2.0;
    let mut ach_d = (opp_cy * opp_cy + opp_gm * opp_gm).max(0.0).sqrt() / SQRT3;
    // Smooth ach_d (CTL 616)
    ach_d = 1.25 * compress_toe_quadratic(ach_d, 0.25);

    // ── Hue angle + RGB/CMY hue windows (CTL 618-634) ──
    // Needed for brilliance (and hue shift in Phase 3)
    let hue = (opp_cy.atan2(opp_gm) + PI + 1.10714931).rem_euclid(2.0 * PI);

    let ha_rgb = [
        gauss_window(hue_offset(hue, 0.1), 0.9),
        gauss_window(hue_offset(hue, 4.3), 0.9),
        gauss_window(hue_offset(hue, 2.3), 0.9),
    ];
    let ha_cmy = [
        gauss_window(hue_offset(hue, 3.3), 0.6),
        gauss_window(hue_offset(hue, 1.3), 0.6),
        gauss_window(hue_offset(hue, -1.2), 0.6),
    ];

    // ── Purity compression range (CTL 637-643) ──
    let mut ts_pt_cmp = 1.0 - spowf(ts_pt, 1.0 / cfg.pt_rng_low);

    let mut pt_rng_high_f = (ach_d / 1.2).min(1.0);
    pt_rng_high_f *= pt_rng_high_f;
    if cfg.pt_rng_high < 1.0 {
        pt_rng_high_f = 1.0 - pt_rng_high_f;
    }

    ts_pt_cmp = spowf(ts_pt_cmp, cfg.pt_rng_high) * (1.0 - pt_rng_high_f)
        + ts_pt_cmp * pt_rng_high_f;

    // ── Brilliance (CTL 646-660) ──
    let mut brl_f: f32 = 1.0;
    if cfg.brl_enable {
        brl_f = -cfg.brl_r * ha_rgb[0] - cfg.brl_g * ha_rgb[1] - cfg.brl_b * ha_rgb[2]
                - cfg.brl_c * ha_cmy[0] - cfg.brl_m * ha_cmy[1] - cfg.brl_y * ha_cmy[2];
        brl_f = (1.0 - ach_d) * brl_f + 1.0 - brl_f;
        brl_f = softplus(brl_f, 0.25, -100.0, 0.0); // Protect against over-darkening

        // Limit by tonescale (CTL 656-658)
        let brl_ts = if brl_f > 1.0 { 1.0 - ts_pt } else { ts_pt };
        let brl_lim = spowf(brl_ts, 1.0 - cfg.brl_rng);
        brl_f = brl_f * brl_lim + 1.0 - brl_lim;
        brl_f = brl_f.clamp(0.0, 2.0); // protect for shadow grain
    }

    // ── Mid-Range Purity (CTL 664-679) ──
    let mut ptm_sc: f32 = 1.0;
    if cfg.ptm_enable {
        // Mid Purity Low (CTL 672-673)
        let ptm_ach_d_low = complement_power(ach_d, cfg.ptm_low_st);
        ptm_sc = sigmoid_cubic(ptm_ach_d_low, cfg.ptm_low * (1.0 - ts_pt));

        // Mid Purity High (CTL 676-677)
        let ptm_ach_d_high = complement_power(ach_d, cfg.ptm_high_st) * (1.0 - ts_pt)
            + ach_d * ach_d * ts_pt;
        ptm_sc *= sigmoid_cubic(ptm_ach_d_high, cfg.ptm_high * ts_pt);
        ptm_sc = ptm_sc.max(0.0);
    }

    // ── Apply brilliance (CTL 724-725) ──
    r *= brl_f;
    g *= brl_f;
    b *= brl_f;

    // ── Apply purity compression + mid purity (CTL 728-729) ──
    ts_pt_cmp *= ptm_sc;
    r = r * ts_pt_cmp + (1.0 - ts_pt_cmp);
    g = g * ts_pt_cmp + (1.0 - ts_pt_cmp);
    b = b * ts_pt_cmp + (1.0 - ts_pt_cmp);

    // ── Inverse rendering space (CTL 731-733) ──
    let sat_l2 = r * cfg.rs_rw + g * rs_gw + b * cfg.rs_bw;
    let inv_sa = 1.0 / (cfg.rs_sa - 1.0);
    r = (sat_l2 * cfg.rs_sa - r) * inv_sa;
    g = (sat_l2 * cfg.rs_sa - g) * inv_sa;
    b = (sat_l2 * cfg.rs_sa - b) * inv_sa;

    // ── Display gamut conversion (CTL 736-752) ──
    // Creative white: cwp=0 (D65) → no-op for both Base and Default
    if cfg.display_gamut == DisplayGamut::Rec709 {
        let t = mat3_apply(&P3D65_TO_REC709, [r, g, b]);
        r = t[0];
        g = t[1];
        b = t[2];
    }
    // Rec.2020: stays in P3 until after clamp

    // ── Purity compress low (CTL 754-763) ──
    if cfg.ptl_enable {
        let sum0 = softplus(r, 0.2, -100.0, -0.3)
            + g
            + softplus(b, 0.2, -100.0, -0.3);
        r = softplus(r, 0.04, -0.3, 0.0);
        g = softplus(g, 0.06, -0.3, 0.0);
        b = softplus(b, 0.01, -0.05, 0.0);
        let total = r + g + b;
        let ptl_norm = sdivf(sum0, total).min(1.0);
        r *= ptl_norm;
        g *= ptl_norm;
        b *= ptl_norm;
    }

    // ── Final tonescale: toe + display scale (CTL 765-771) ──
    tsn *= ts.ts_m2;
    tsn = compress_toe_quadratic(tsn, cfg.tn_toe);
    tsn *= ts.ts_dsc;

    // Return from RGB ratios to absolute values
    r *= tsn;
    g *= tsn;
    b *= tsn;

    // Clamp [0, 1]
    r = r.clamp(0.0, 1.0);
    g = g.clamp(0.0, 1.0);
    b = b.clamp(0.0, 1.0);

    // Rec.2020: P3 → Rec.2020 (CTL 777-781)
    if cfg.display_gamut == DisplayGamut::Rec2020 {
        r = r.max(0.0);
        g = g.max(0.0);
        b = b.max(0.0);
        let t = mat3_apply(&P3D65_TO_REC2020, [r, g, b]);
        return [t[0], t[1], t[2]];
    }

    [r, g, b]
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Process from P3-D65 input (identity cam_to_p3 matrix).
    fn process_p3(rgb: [f32; 3], cfg: &OpenDrtConfig, ts: &TonescaleParams) -> [f32; 3] {
        let identity: Mat3 = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        process_pixel(rgb, ts, cfg, &identity)
    }

    fn run_vectors(preset_filter: Option<&str>) {
        let data = include_str!("../../../tests/opendrt/opendrt_test_vectors.json");
        let parsed: serde_json::Value = serde_json::from_str(data).expect("parse test vectors");
        let tolerance: f64 = parsed["tolerance"].as_f64().unwrap();
        let vectors = parsed["vectors"].as_array().unwrap();

        let mut max_err: f64 = 0.0;
        let mut fail_count = 0;
        let mut tested = 0;

        for v in vectors {
            let name = v["name"].as_str().unwrap();

            // Filter by preset if specified
            if let Some(filter) = preset_filter {
                let preset = v["preset"].as_str().unwrap_or("base");
                if preset != filter { continue; }
            }

            let input: Vec<f32> = v["input"].as_array().unwrap()
                .iter().map(|x| x.as_f64().unwrap() as f32).collect();
            let expected: Vec<f64> = v["expected"].as_array().unwrap()
                .iter().map(|x| x.as_f64().unwrap()).collect();
            let tn_lp = v["tn_Lp"].as_f64().unwrap() as f32;
            let tn_gb = v["tn_gb"].as_f64().unwrap() as f32;
            let pt_hdr = v["pt_hdr"].as_f64().unwrap() as f32;
            let display_gamut = v["display_gamut"].as_i64().unwrap();
            let preset = v["preset"].as_str().unwrap_or("base");

            let gamut = match display_gamut {
                0 => DisplayGamut::Rec709,
                2 => DisplayGamut::Rec2020,
                _ => panic!("unknown display gamut {display_gamut}"),
            };

            let cfg = {
                let base = match preset {
                    "default" => OpenDrtConfig::default_sdr(),
                    _ => OpenDrtConfig::base_sdr(),
                };
                OpenDrtConfig {
                    peak_luminance: tn_lp,
                    grey_boost: tn_gb,
                    pt_hdr,
                    display_gamut: gamut,
                    ..base
                }
            };
            let ts = TonescaleParams::new(&cfg);
            let result = process_p3([input[0], input[1], input[2]], &cfg, &ts);

            for ch in 0..3 {
                let err = (result[ch] as f64 - expected[ch]).abs();
                max_err = max_err.max(err);
                if err > tolerance {
                    eprintln!(
                        "FAIL {name} ch{ch}: got {:.8}, expected {:.8}, err {:.2e}",
                        result[ch], expected[ch], err
                    );
                    fail_count += 1;
                }
            }
            tested += 1;
        }

        eprintln!("Tested {tested} vectors, max error: {max_err:.2e}");
        assert!(tested > 0, "no vectors matched filter");
        assert_eq!(fail_count, 0, "{fail_count} channel(s) exceeded tolerance {tolerance:.0e}");
    }

    #[test]
    fn test_golden_vectors_base() {
        run_vectors(Some("base"));
    }

    #[test]
    fn test_golden_vectors_default() {
        run_vectors(Some("default"));
    }

    #[test]
    fn test_grey_018_sdr_maps_to_lg() {
        let cfg = OpenDrtConfig::base_sdr();
        let ts = TonescaleParams::new(&cfg);
        let result = process_p3([0.18, 0.18, 0.18], &cfg, &ts);
        // 0.18 scene → tn_Lg/100 = 0.111 display-linear
        let expected = cfg.tn_lg / 100.0;
        assert!((result[0] - expected).abs() < 1e-4,
            "grey 0.18 SDR: got {:.6}, expected {:.6}", result[0], expected);
        // R = G = B for grey
        assert!((result[0] - result[1]).abs() < 1e-6);
        assert!((result[1] - result[2]).abs() < 1e-6);
    }
}
