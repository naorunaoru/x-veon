#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 fragColor;

// ── Uniforms ────────────────────────────────────────────────────────────
uniform sampler2D u_image;
uniform int u_orientation;       // 0=Normal, 1=Rot90, 2=Rot180, 3=Rot270
uniform int u_toneMapMode;       // 0=legacy, 1=opendrt
uniform float u_legacyPeakScale;

// Tonescale precomputed params
uniform float u_ts_s;
uniform float u_ts_s1;
uniform float u_ts_m2;
uniform float u_ts_dsc;
uniform float u_ts_x0;

// OpenDRT config packed into vec4s
uniform vec4 u_odrt_tone;       // (tn_con, tn_sh, tn_toe, tn_off)
uniform vec4 u_odrt_rs;         // (rs_sa, rs_rw, rs_bw, 0)
uniform vec4 u_odrt_pt;         // (pt_r, pt_g, pt_b, pt_rng_low)
uniform vec4 u_odrt_pt2;        // (pt_rng_high, 0, 0, 0)
uniform vec4 u_odrt_lcon;       // (enable, tn_lcon, tn_lcon_w, tn_lcon_pc)
uniform vec4 u_odrt_hcon;       // (enable, tn_hcon, tn_hcon_pv, tn_hcon_st)
uniform vec4 u_odrt_brl_rgb;    // (brl_r, brl_g, brl_b, brl_rng)
uniform vec4 u_odrt_brl_cmy;    // (brl_c, brl_m, brl_y, enable)
uniform vec4 u_odrt_ptm;        // (enable, ptm_low, ptm_low_st, ptm_high)
uniform float u_odrt_ptm_high_st;
uniform float u_odrt_ptl_enable;

uniform mat3 u_srgbToP3;
uniform mat3 u_p3ToDisplay;

// ── Constants ───────────────────────────────────────────────────────────
const float PI = 3.14159265;
const float SQRT3 = 1.7320508;

// ── Math helpers (1:1 port of opendrt.rs) ───────────────────────────────

float spowf(float x, float p) {
  return x <= 0.0 ? x : pow(x, p);
}

float sdivf(float a, float b) {
  return b == 0.0 ? 0.0 : a / b;
}

float compress_hp(float x, float s, float p) {
  return spowf(x / (x + s), p);
}

float compress_toe_quadratic(float x, float toe) {
  if (toe == 0.0) return x;
  return spowf(x, 2.0) / (x + toe);
}

float compress_toe_cubic_fwd(float x, float m, float w) {
  if (m == 1.0) return x;
  float x2 = x * x;
  return x * (x2 + m * w) / (x2 + w);
}

float compress_toe_cubic_inv(float x, float m, float w) {
  if (m == 1.0) return x;
  float x2 = x * x;
  float p0 = x2 - 3.0 * m * w;
  float p1 = 2.0 * x2 + 27.0 * w - 9.0 * m * w;
  float disc = max(x2 * p1 * p1 - 4.0 * p0 * p0 * p0, 0.0);
  float p2 = pow(sqrt(disc) * 0.5 + x * p1 * 0.5, 1.0 / 3.0);
  return p0 / (3.0 * p2) + p2 / 3.0 + x / 3.0;
}

float contrast_high(float x, float p, float pv, float pv_lx) {
  float x0 = 0.18 * pow(2.0, pv);
  if (x < x0 || p == 1.0) return x;
  float o = x0 - x0 / p;
  float s0 = pow(x0, 1.0 - p) / p;
  float x1 = x0 * pow(2.0, pv_lx);
  float k1 = p * s0 * pow(x1, p) / x1;
  float y1 = s0 * pow(x1, p) + o;
  return x > x1 ? k1 * (x - x1) + y1 : s0 * pow(x, p) + o;
}

float complement_power(float x, float p) {
  return 1.0 - spowf(1.0 - x, 1.0 / p);
}

float sigmoid_cubic(float x, float s) {
  if (x < 0.0 || x > 1.0) return 1.0;
  return 1.0 + s * (1.0 - 3.0 * x * x + 2.0 * x * x * x);
}

float gauss_window(float x, float w) {
  float y = x / w;
  return exp(-y * y);
}

float hue_offset(float h, float o) {
  return mod(h - o + PI, 2.0 * PI) - PI;
}

float softplus(float x, float s, float x0, float y0) {
  if (x > 10.0 * s + y0 || s < 0.001) return x;
  float m = 1.0;
  if (abs(y0) > 1e-6) m = exp(y0 / s);
  m -= exp(x0 / s);
  return s * log(max(0.0, m + exp(x / s)));
}

// ── sRGB OETF ───────────────────────────────────────────────────────────

float srgb_oetf(float v) {
  v = clamp(v, 0.0, 1.0);
  return v <= 0.0031308 ? v * 12.92 : 1.055 * pow(v, 1.0 / 2.4) - 0.055;
}

// ── OpenDRT core (1:1 port of process_pixel from opendrt.rs) ────────────

vec3 opendrt(vec3 rgb_in) {
  // Unpack uniforms
  float tn_con = u_odrt_tone.x;
  float tn_toe = u_odrt_tone.z;
  float tn_off = u_odrt_tone.w;
  float rs_sa  = u_odrt_rs.x;
  float rs_rw  = u_odrt_rs.y;
  float rs_bw  = u_odrt_rs.z;
  float pt_r   = u_odrt_pt.x;
  float pt_g   = u_odrt_pt.y;
  float pt_b   = u_odrt_pt.z;
  float pt_rng_low  = u_odrt_pt.w;
  float pt_rng_high = u_odrt_pt2.x;

  // Stage 1: sRGB → P3-D65
  vec3 p3 = u_srgbToP3 * rgb_in;
  float r = p3.x, g = p3.y, b = p3.z;

  // Stage 2: Rendering space saturation + offset
  float rs_gw = 1.0 - rs_rw - rs_bw;
  float sat_l = r * rs_rw + g * rs_gw + b * rs_bw;
  r = sat_l * rs_sa + r * (1.0 - rs_sa);
  g = sat_l * rs_sa + g * (1.0 - rs_sa);
  b = sat_l * rs_sa + b * (1.0 - rs_sa);
  r += tn_off; g += tn_off; b += tn_off;

  // Contrast Low
  if (u_odrt_lcon.x > 0.5) {
    float tn_lcon = u_odrt_lcon.y;
    float tn_lcon_w = u_odrt_lcon.z;
    float tn_lcon_pc = u_odrt_lcon.w;

    float mcon_m = exp2(-tn_lcon);
    float mcon_w_half = tn_lcon_w / 4.0;
    float mcon_w = mcon_w_half * mcon_w_half;

    float mcon_cnst_sc = compress_toe_cubic_inv(u_ts_x0, mcon_m, mcon_w) / u_ts_x0;
    r *= mcon_cnst_sc;
    g *= mcon_cnst_sc;
    b *= mcon_cnst_sc;

    float mr = max(r, 0.0), mg = max(g, 0.0), mb = max(b, 0.0);
    float mcon_nm = sqrt(mr * mr + mg * mg + mb * mb) / SQRT3;
    float mcon_sc = (mcon_nm * mcon_nm + mcon_m * mcon_w) / (mcon_nm * mcon_nm + mcon_w);

    if (tn_lcon_pc > 0.0) {
      float mcon_r = compress_toe_cubic_fwd(r, mcon_m, mcon_w);
      float mcon_g = compress_toe_cubic_fwd(g, mcon_m, mcon_w);
      float mcon_b = compress_toe_cubic_fwd(b, mcon_m, mcon_w);

      float mcon_mx = max(r, max(g, b));
      float mcon_mn = min(r, min(g, b));
      float mcon_ch = clamp(1.0 - sdivf(mcon_mn, mcon_mx), 0.0, 1.0);
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
  float rc = max(r, 0.0), gc = max(g, 0.0), bc = max(b, 0.0);
  float tsn = sqrt(rc * rc + gc * gc + bc * bc) / SQRT3;

  // Purity compression norm
  float ts_pt = sqrt(max(r * r * pt_r + g * g * pt_g + b * b * pt_b, 0.0));

  // RGB ratios
  if (tsn > 0.0) {
    float inv = 1.0 / tsn;
    r = max(r, -2.0) * inv;
    g = max(g, -2.0) * inv;
    b = max(b, -2.0) * inv;
  } else {
    r = 0.0; g = 0.0; b = 0.0;
  }

  // Contrast High
  if (u_odrt_hcon.x > 0.5) {
    float hcon_p = exp2(u_odrt_hcon.y);
    tsn = contrast_high(tsn, hcon_p, u_odrt_hcon.z, u_odrt_hcon.w);
    ts_pt = contrast_high(ts_pt, hcon_p, u_odrt_hcon.z, u_odrt_hcon.w);
  }

  // Apply tonescale
  tsn = compress_hp(tsn, u_ts_s, tn_con);
  ts_pt = compress_hp(ts_pt, u_ts_s1, tn_con);

  // Opponent space / achromatic distance
  float opp_cy = r - b;
  float opp_gm = g - (r + b) / 2.0;
  float ach_d = sqrt(max(opp_cy * opp_cy + opp_gm * opp_gm, 0.0)) / SQRT3;
  ach_d = 1.25 * compress_toe_quadratic(ach_d, 0.25);

  // Hue angle + RGB/CMY hue windows
  float hue = mod(atan(opp_cy, opp_gm) + PI + 1.10714931, 2.0 * PI);

  float ha_r = gauss_window(hue_offset(hue, 0.1), 0.9);
  float ha_g = gauss_window(hue_offset(hue, 4.3), 0.9);
  float ha_b = gauss_window(hue_offset(hue, 2.3), 0.9);
  float ha_c = gauss_window(hue_offset(hue, 3.3), 0.6);
  float ha_m = gauss_window(hue_offset(hue, 1.3), 0.6);
  float ha_y = gauss_window(hue_offset(hue, -1.2), 0.6);

  // Purity compression range
  float ts_pt_cmp = 1.0 - spowf(ts_pt, 1.0 / pt_rng_low);

  float pt_rng_high_f = min(ach_d / 1.2, 1.0);
  pt_rng_high_f *= pt_rng_high_f;
  if (pt_rng_high < 1.0) {
    pt_rng_high_f = 1.0 - pt_rng_high_f;
  }

  ts_pt_cmp = spowf(ts_pt_cmp, pt_rng_high) * (1.0 - pt_rng_high_f)
            + ts_pt_cmp * pt_rng_high_f;

  // Brilliance
  float brl_f = 1.0;
  if (u_odrt_brl_cmy.w > 0.5) {
    float brl_r = u_odrt_brl_rgb.x;
    float brl_g = u_odrt_brl_rgb.y;
    float brl_b = u_odrt_brl_rgb.z;
    float brl_rng = u_odrt_brl_rgb.w;
    float brl_c = u_odrt_brl_cmy.x;
    float brl_m = u_odrt_brl_cmy.y;
    float brl_y = u_odrt_brl_cmy.z;

    brl_f = -brl_r * ha_r - brl_g * ha_g - brl_b * ha_b
            - brl_c * ha_c - brl_m * ha_m - brl_y * ha_y;
    brl_f = (1.0 - ach_d) * brl_f + 1.0 - brl_f;
    brl_f = softplus(brl_f, 0.25, -100.0, 0.0);

    float brl_ts = brl_f > 1.0 ? 1.0 - ts_pt : ts_pt;
    float brl_lim = spowf(brl_ts, 1.0 - brl_rng);
    brl_f = brl_f * brl_lim + 1.0 - brl_lim;
    brl_f = clamp(brl_f, 0.0, 2.0);
  }

  // Mid-Range Purity
  float ptm_sc = 1.0;
  if (u_odrt_ptm.x > 0.5) {
    float ptm_low = u_odrt_ptm.y;
    float ptm_low_st = u_odrt_ptm.z;
    float ptm_high = u_odrt_ptm.w;
    float ptm_high_st = u_odrt_ptm_high_st;

    float ptm_ach_d_low = complement_power(ach_d, ptm_low_st);
    ptm_sc = sigmoid_cubic(ptm_ach_d_low, ptm_low * (1.0 - ts_pt));

    float ptm_ach_d_high = complement_power(ach_d, ptm_high_st) * (1.0 - ts_pt)
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
  float sat_l2 = r * rs_rw + g * rs_gw + b * rs_bw;
  float inv_sa = 1.0 / (rs_sa - 1.0);
  r = (sat_l2 * rs_sa - r) * inv_sa;
  g = (sat_l2 * rs_sa - g) * inv_sa;
  b = (sat_l2 * rs_sa - b) * inv_sa;

  // Display gamut conversion (P3 → Rec.709 for SDR).
  // NOTE: For Rec.2020 this should happen AFTER clamp (see opendrt.rs lines 585-592).
  // Currently SDR-only, so this position is correct for Rec.709.
  vec3 disp = u_p3ToDisplay * vec3(r, g, b);
  r = disp.x; g = disp.y; b = disp.z;

  // Purity compress low
  if (u_odrt_ptl_enable > 0.5) {
    float sum0 = softplus(r, 0.2, -100.0, -0.3)
               + g
               + softplus(b, 0.2, -100.0, -0.3);
    r = softplus(r, 0.04, -0.3, 0.0);
    g = softplus(g, 0.06, -0.3, 0.0);
    b = softplus(b, 0.01, -0.05, 0.0);
    float total = r + g + b;
    float ptl_norm = min(sdivf(sum0, total), 1.0);
    r *= ptl_norm;
    g *= ptl_norm;
    b *= ptl_norm;
  }

  // Final tonescale: toe + display scale
  tsn *= u_ts_m2;
  tsn = compress_toe_quadratic(tsn, tn_toe);
  tsn *= u_ts_dsc;

  // Return from RGB ratios to absolute values
  r *= tsn;
  g *= tsn;
  b *= tsn;

  return clamp(vec3(r, g, b), 0.0, 1.0);
}

// ── EXIF rotation (remap UV) ────────────────────────────────────────────

vec2 rotateUV(vec2 uv) {
  if (u_orientation == 1) return vec2(1.0 - uv.y, uv.x);       // Rotate90
  if (u_orientation == 2) return vec2(1.0 - uv.x, 1.0 - uv.y); // Rotate180
  if (u_orientation == 3) return vec2(uv.y, 1.0 - uv.x);       // Rotate270
  return uv;
}

// ── Main ────────────────────────────────────────────────────────────────

void main() {
  vec2 texUV = rotateUV(v_uv);
  vec3 linear = texture(u_image, texUV).rgb;

  vec3 display;
  if (u_toneMapMode == 1) {
    display = opendrt(linear);
  } else {
    // Legacy: scale super-whites, clamp to [0,1]
    display = clamp(linear * u_legacyPeakScale, 0.0, 1.0);
  }

  // sRGB OETF
  fragColor = vec4(srgb_oetf(display.r), srgb_oetf(display.g), srgb_oetf(display.b), 1.0);
}
