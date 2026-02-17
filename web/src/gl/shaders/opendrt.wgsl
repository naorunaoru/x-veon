// OpenDRT tone mapping shader — WGSL port of frag.glsl + vert.glsl.
// Combined vertex + fragment in a single module.

// ── Uniforms ────────────────────────────────────────────────────────────

struct Uniforms {
  ts: vec4f,                  // (ts_s, ts_s1, ts_m2, ts_dsc)
  ts_x0_and_flags: vec4f,    // (ts_x0, hdrDisplay, exportMode, ptl_enable)
  odrt_tone: vec4f,           // (tn_con, tn_sh, tn_toe, tn_off)
  odrt_rs: vec4f,             // (rs_sa, rs_rw, rs_bw, 0)
  odrt_pt: vec4f,             // (pt_r, pt_g, pt_b, pt_rng_low)
  odrt_pt2: vec4f,            // (pt_rng_high, ptm_high_st, 0, 0)
  odrt_lcon: vec4f,           // (enable, tn_lcon, tn_lcon_w, tn_lcon_pc)
  odrt_hcon: vec4f,           // (enable, tn_hcon, tn_hcon_pv, tn_hcon_st)
  odrt_brl_rgb: vec4f,        // (brl_r, brl_g, brl_b, brl_rng)
  odrt_brl_cmy: vec4f,        // (brl_c, brl_m, brl_y, enable)
  odrt_ptm: vec4f,            // (enable, ptm_low, ptm_low_st, ptm_high)
  preprocess: vec4f,          // (exposure, wb_temp, wb_tint, sharpen_amount)
  texel_size_pad: vec4f,      // (1/w, 1/h, 0, 0)
  srgb_to_p3_col0: vec4f,
  srgb_to_p3_col1: vec4f,
  srgb_to_p3_col2: vec4f,
  p3_to_display_col0: vec4f,
  p3_to_display_col1: vec4f,
  p3_to_display_col2: vec4f,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var t_image: texture_2d<f32>;
@group(0) @binding(2) var s_image: sampler;

// ── Constants ───────────────────────────────────────────────────────────

const PI: f32 = 3.14159265;
const SQRT3: f32 = 1.7320508;

// ── Vertex ──────────────────────────────────────────────────────────────

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOutput {
  let x = f32((vid & 1u) << 2u) - 1.0;
  let y = f32((vid & 2u) << 1u) - 1.0;
  var out: VertexOutput;
  out.position = vec4f(x, y, 0.0, 1.0);
  out.uv = vec2f((x + 1.0) * 0.5, 1.0 - (y + 1.0) * 0.5);
  return out;
}

// ── Math helpers (1:1 port of opendrt.rs) ───────────────────────────────

fn spowf(x: f32, p: f32) -> f32 {
  if (x <= 0.0) { return x; }
  return pow(x, p);
}

fn sdivf(a: f32, b: f32) -> f32 {
  if (b == 0.0) { return 0.0; }
  return a / b;
}

fn compress_hp(x: f32, s: f32, p: f32) -> f32 {
  return spowf(x / (x + s), p);
}

fn compress_toe_quadratic(x: f32, toe: f32) -> f32 {
  if (toe == 0.0) { return x; }
  return spowf(x, 2.0) / (x + toe);
}

fn compress_toe_cubic_fwd(x: f32, m: f32, w: f32) -> f32 {
  if (m == 1.0) { return x; }
  let x2 = x * x;
  return x * (x2 + m * w) / (x2 + w);
}

fn compress_toe_cubic_inv(x: f32, m: f32, w: f32) -> f32 {
  if (m == 1.0) { return x; }
  let x2 = x * x;
  let p0 = x2 - 3.0 * m * w;
  let p1 = 2.0 * x2 + 27.0 * w - 9.0 * m * w;
  let disc = max(x2 * p1 * p1 - 4.0 * p0 * p0 * p0, 0.0);
  let p2 = pow(sqrt(disc) * 0.5 + x * p1 * 0.5, 1.0 / 3.0);
  return p0 / (3.0 * p2) + p2 / 3.0 + x / 3.0;
}

fn contrast_high(x: f32, p: f32, pv: f32, pv_lx: f32) -> f32 {
  let x0 = 0.18 * pow(2.0, pv);
  if (x < x0 || p == 1.0) { return x; }
  let o = x0 - x0 / p;
  let s0 = pow(x0, 1.0 - p) / p;
  let x1 = x0 * pow(2.0, pv_lx);
  let k1 = p * s0 * pow(x1, p) / x1;
  let y1 = s0 * pow(x1, p) + o;
  if (x > x1) { return k1 * (x - x1) + y1; }
  return s0 * pow(x, p) + o;
}

fn complement_power(x: f32, p: f32) -> f32 {
  return 1.0 - spowf(1.0 - x, 1.0 / p);
}

fn sigmoid_cubic(x: f32, s: f32) -> f32 {
  if (x < 0.0 || x > 1.0) { return 1.0; }
  return 1.0 + s * (1.0 - 3.0 * x * x + 2.0 * x * x * x);
}

fn gauss_window(x: f32, w: f32) -> f32 {
  let y = x / w;
  return exp(-y * y);
}

// GLSL mod(a, b) = a - b * floor(a / b) — WGSL has no float mod
fn fmod(a: f32, b: f32) -> f32 {
  return a - b * floor(a / b);
}

fn hue_offset(h: f32, o: f32) -> f32 {
  return fmod(h - o + PI, 2.0 * PI) - PI;
}

fn softplus(x: f32, s: f32, x0: f32, y0: f32) -> f32 {
  if (x > 10.0 * s + y0 || s < 0.001) { return x; }
  var m: f32 = 1.0;
  if (abs(y0) > 1e-6) { m = exp(y0 / s); }
  m -= exp(x0 / s);
  return s * log(max(0.0, m + exp(x / s)));
}

// ── sRGB OETF ───────────────────────────────────────────────────────────

fn srgb_oetf(v_in: f32) -> f32 {
  var v: f32;
  if (u.ts_x0_and_flags.y < 0.5) {
    v = clamp(v_in, 0.0, 1.0);
  } else {
    v = max(v_in, 0.0);
  }
  if (v <= 0.0031308) { return v * 12.92; }
  return 1.055 * pow(v, 1.0 / 2.4) - 0.055;
}

// ── OpenDRT core (1:1 port of process_pixel from opendrt.rs) ────────────

fn opendrt(rgb_in: vec3f) -> vec3f {
  // Unpack uniforms
  let tn_con = u.odrt_tone.x;
  let tn_toe = u.odrt_tone.z;
  let tn_off = u.odrt_tone.w;
  let rs_sa  = u.odrt_rs.x;
  let rs_rw  = u.odrt_rs.y;
  let rs_bw  = u.odrt_rs.z;
  let pt_r   = u.odrt_pt.x;
  let pt_g   = u.odrt_pt.y;
  let pt_b   = u.odrt_pt.z;
  let pt_rng_low  = u.odrt_pt.w;
  let pt_rng_high = u.odrt_pt2.x;

  // Reconstruct matrices from column vec4s
  let srgb_to_p3 = mat3x3f(
    u.srgb_to_p3_col0.xyz,
    u.srgb_to_p3_col1.xyz,
    u.srgb_to_p3_col2.xyz,
  );
  let p3_to_display = mat3x3f(
    u.p3_to_display_col0.xyz,
    u.p3_to_display_col1.xyz,
    u.p3_to_display_col2.xyz,
  );

  // Stage 1: sRGB -> P3-D65
  let p3 = srgb_to_p3 * rgb_in;
  var r = p3.x;
  var g = p3.y;
  var b = p3.z;

  // Stage 2: Rendering space saturation + offset
  let rs_gw = 1.0 - rs_rw - rs_bw;
  var sat_l = r * rs_rw + g * rs_gw + b * rs_bw;
  r = sat_l * rs_sa + r * (1.0 - rs_sa);
  g = sat_l * rs_sa + g * (1.0 - rs_sa);
  b = sat_l * rs_sa + b * (1.0 - rs_sa);
  r += tn_off;
  g += tn_off;
  b += tn_off;

  // Contrast Low
  if (u.odrt_lcon.x > 0.5) {
    let tn_lcon = u.odrt_lcon.y;
    let tn_lcon_w = u.odrt_lcon.z;
    let tn_lcon_pc = u.odrt_lcon.w;

    let mcon_m = exp2(-tn_lcon);
    let mcon_w_half = tn_lcon_w / 4.0;
    let mcon_w = mcon_w_half * mcon_w_half;

    let mcon_cnst_sc = compress_toe_cubic_inv(u.ts_x0_and_flags.x, mcon_m, mcon_w) / u.ts_x0_and_flags.x;
    r *= mcon_cnst_sc;
    g *= mcon_cnst_sc;
    b *= mcon_cnst_sc;

    let mr = max(r, 0.0);
    let mg = max(g, 0.0);
    let mb = max(b, 0.0);
    let mcon_nm = sqrt(mr * mr + mg * mg + mb * mb) / SQRT3;
    let mcon_sc = (mcon_nm * mcon_nm + mcon_m * mcon_w) / (mcon_nm * mcon_nm + mcon_w);

    if (tn_lcon_pc > 0.0) {
      let mcon_r = compress_toe_cubic_fwd(r, mcon_m, mcon_w);
      let mcon_g = compress_toe_cubic_fwd(g, mcon_m, mcon_w);
      let mcon_b = compress_toe_cubic_fwd(b, mcon_m, mcon_w);

      let mcon_mx = max(r, max(g, b));
      let mcon_mn = min(r, min(g, b));
      var mcon_ch = clamp(1.0 - sdivf(mcon_mn, mcon_mx), 0.0, 1.0);
      mcon_ch = pow(mcon_ch, 4.0 * tn_lcon_pc);
      r = r * mcon_sc * mcon_ch + mcon_r * (1.0 - mcon_ch);
      g = g * mcon_sc * mcon_ch + mcon_g * (1.0 - mcon_ch);
      b = b * mcon_sc * mcon_ch + mcon_b * (1.0 - mcon_ch);
    } else {
      r *= mcon_sc;
      g *= mcon_sc;
      b *= mcon_sc;
    }
  }

  // Tonescale norm
  let rc = max(r, 0.0);
  let gc = max(g, 0.0);
  let bc = max(b, 0.0);
  var tsn = sqrt(rc * rc + gc * gc + bc * bc) / SQRT3;

  // Purity compression norm
  var ts_pt = sqrt(max(r * r * pt_r + g * g * pt_g + b * b * pt_b, 0.0));

  // RGB ratios
  if (tsn > 0.0) {
    let inv = 1.0 / tsn;
    r = max(r, -2.0) * inv;
    g = max(g, -2.0) * inv;
    b = max(b, -2.0) * inv;
  } else {
    r = 0.0;
    g = 0.0;
    b = 0.0;
  }

  // Contrast High
  if (u.odrt_hcon.x > 0.5) {
    let hcon_p = exp2(u.odrt_hcon.y);
    tsn = contrast_high(tsn, hcon_p, u.odrt_hcon.z, u.odrt_hcon.w);
    ts_pt = contrast_high(ts_pt, hcon_p, u.odrt_hcon.z, u.odrt_hcon.w);
  }

  // Apply tonescale
  tsn = compress_hp(tsn, u.ts.x, tn_con);
  ts_pt = compress_hp(ts_pt, u.ts.y, tn_con);

  // Opponent space / achromatic distance
  let opp_cy = r - b;
  let opp_gm = g - (r + b) / 2.0;
  var ach_d = sqrt(max(opp_cy * opp_cy + opp_gm * opp_gm, 0.0)) / SQRT3;
  ach_d = 1.25 * compress_toe_quadratic(ach_d, 0.25);

  // Hue angle + RGB/CMY hue windows
  let hue = fmod(atan2(opp_cy, opp_gm) + PI + 1.10714931, 2.0 * PI);

  let ha_r = gauss_window(hue_offset(hue, 0.1), 0.9);
  let ha_g = gauss_window(hue_offset(hue, 4.3), 0.9);
  let ha_b = gauss_window(hue_offset(hue, 2.3), 0.9);
  let ha_c = gauss_window(hue_offset(hue, 3.3), 0.6);
  let ha_m = gauss_window(hue_offset(hue, 1.3), 0.6);
  let ha_y = gauss_window(hue_offset(hue, -1.2), 0.6);

  // Purity compression range
  var ts_pt_cmp = 1.0 - spowf(ts_pt, 1.0 / pt_rng_low);

  var pt_rng_high_f = min(ach_d / 1.2, 1.0);
  pt_rng_high_f *= pt_rng_high_f;
  if (pt_rng_high < 1.0) {
    pt_rng_high_f = 1.0 - pt_rng_high_f;
  }

  ts_pt_cmp = spowf(ts_pt_cmp, pt_rng_high) * (1.0 - pt_rng_high_f)
            + ts_pt_cmp * pt_rng_high_f;

  // Brilliance
  var brl_f: f32 = 1.0;
  if (u.odrt_brl_cmy.w > 0.5) {
    let brl_r = u.odrt_brl_rgb.x;
    let brl_g = u.odrt_brl_rgb.y;
    let brl_b = u.odrt_brl_rgb.z;
    let brl_rng = u.odrt_brl_rgb.w;
    let brl_c = u.odrt_brl_cmy.x;
    let brl_m = u.odrt_brl_cmy.y;
    let brl_y = u.odrt_brl_cmy.z;

    brl_f = -brl_r * ha_r - brl_g * ha_g - brl_b * ha_b
            - brl_c * ha_c - brl_m * ha_m - brl_y * ha_y;
    brl_f = (1.0 - ach_d) * brl_f + 1.0 - brl_f;
    brl_f = softplus(brl_f, 0.25, -100.0, 0.0);

    var brl_ts: f32;
    if (brl_f > 1.0) { brl_ts = 1.0 - ts_pt; } else { brl_ts = ts_pt; }
    let brl_lim = spowf(brl_ts, 1.0 - brl_rng);
    brl_f = brl_f * brl_lim + 1.0 - brl_lim;
    brl_f = clamp(brl_f, 0.0, 2.0);
  }

  // Mid-Range Purity
  var ptm_sc: f32 = 1.0;
  if (u.odrt_ptm.x > 0.5) {
    let ptm_low = u.odrt_ptm.y;
    let ptm_low_st = u.odrt_ptm.z;
    let ptm_high = u.odrt_ptm.w;
    let ptm_high_st = u.odrt_pt2.y;

    let ptm_ach_d_low = complement_power(ach_d, ptm_low_st);
    ptm_sc = sigmoid_cubic(ptm_ach_d_low, ptm_low * (1.0 - ts_pt));

    let ptm_ach_d_high = complement_power(ach_d, ptm_high_st) * (1.0 - ts_pt)
                       + ach_d * ach_d * ts_pt;
    ptm_sc *= sigmoid_cubic(ptm_ach_d_high, ptm_high * ts_pt);
    ptm_sc = max(ptm_sc, 0.0);
  }

  // Apply brilliance
  r *= brl_f;
  g *= brl_f;
  b *= brl_f;

  // Apply purity compression + mid purity
  ts_pt_cmp *= ptm_sc;
  r = r * ts_pt_cmp + (1.0 - ts_pt_cmp);
  g = g * ts_pt_cmp + (1.0 - ts_pt_cmp);
  b = b * ts_pt_cmp + (1.0 - ts_pt_cmp);

  // Inverse rendering space
  let sat_l2 = r * rs_rw + g * rs_gw + b * rs_bw;
  let inv_sa = 1.0 / (min(rs_sa, 0.999) - 1.0);
  r = (sat_l2 * rs_sa - r) * inv_sa;
  g = (sat_l2 * rs_sa - g) * inv_sa;
  b = (sat_l2 * rs_sa - b) * inv_sa;

  // Display gamut conversion (P3 -> Rec.709 for SDR)
  let disp = p3_to_display * vec3f(r, g, b);
  r = disp.x;
  g = disp.y;
  b = disp.z;

  // Purity compress low
  if (u.ts_x0_and_flags.w > 0.5) {
    let sum0 = softplus(r, 0.2, -100.0, -0.3)
             + g
             + softplus(b, 0.2, -100.0, -0.3);
    r = softplus(r, 0.04, -0.3, 0.0);
    g = softplus(g, 0.06, -0.3, 0.0);
    b = softplus(b, 0.01, -0.05, 0.0);
    let total = r + g + b;
    let ptl_norm = min(sdivf(sum0, total), 1.0);
    r *= ptl_norm;
    g *= ptl_norm;
    b *= ptl_norm;
  }

  // Final tonescale: toe + display scale
  tsn *= u.ts.z;
  tsn = compress_toe_quadratic(tsn, tn_toe);
  tsn *= u.ts.w;

  // Return from RGB ratios to absolute values
  r *= tsn;
  g *= tsn;
  b *= tsn;

  if (u.ts_x0_and_flags.y < 0.5) {
    return clamp(vec3f(r, g, b), vec3f(0.0), vec3f(1.0));
  }
  return max(vec3f(r, g, b), vec3f(0.0));
}

// ── Fragment ────────────────────────────────────────────────────────────

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
  var linear = textureSample(t_image, s_image, in.uv).rgb;

  let sharpen_amount = u.preprocess.w;
  let texel_size = u.texel_size_pad.xy;

  // Unsharp mask sharpening
  if (sharpen_amount > 0.0) {
    let t = textureSample(t_image, s_image, in.uv + vec2f(0.0, texel_size.y)).rgb;
    let b = textureSample(t_image, s_image, in.uv - vec2f(0.0, texel_size.y)).rgb;
    let l = textureSample(t_image, s_image, in.uv - vec2f(texel_size.x, 0.0)).rgb;
    let r = textureSample(t_image, s_image, in.uv + vec2f(texel_size.x, 0.0)).rgb;
    let blur = (t + b + l + r) * 0.25;
    linear = max(linear + sharpen_amount * (linear - blur), vec3f(0.0));
  }

  // Pre-processing: exposure + white balance correction
  let exposure = u.preprocess.x;
  let wb_temp = u.preprocess.y;
  let wb_tint = u.preprocess.z;

  linear *= exp2(exposure);
  linear.x *= exp2(wb_temp);
  linear.y *= exp2(-wb_tint);
  linear.z *= exp2(-wb_temp);
  // Compensate tint luminance shift
  linear /= 0.2126 * exp2(wb_temp) + 0.7152 * exp2(-wb_tint) + 0.0722 * exp2(-wb_temp);

  let display = opendrt(linear);

  if (u.ts_x0_and_flags.z > 0.5) {
    // Export: output display-linear (no OETF, no clamp)
    return vec4f(display, 1.0);
  }
  // Display: apply sRGB OETF
  return vec4f(srgb_oetf(display.r), srgb_oetf(display.g), srgb_oetf(display.b), 1.0);
}
