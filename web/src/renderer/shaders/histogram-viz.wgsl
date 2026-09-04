// Fullscreen-triangle fragment shader that reads histogram bins and draws
// the visualization directly on the GPU. No CPU readback needed.

struct VizParams {
  bin_lo: f32,
  bin_hi: f32,
  max_log: f32,
  pad0: f32,
};

struct VizConfig {
  canvas_w: f32,
  canvas_h: f32,
  mode: f32,       // 0 = rgb, 1 = luma, 2 = ev
  clip_bin: f32,    // bin index at value 1.0 (for HDR overlay in linear modes)
};

@group(0) @binding(0) var<storage, read> bins: array<u32, 1024>;
@group(0) @binding(1) var<storage, read> params: VizParams;
@group(0) @binding(2) var<uniform> config: VizConfig;

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

// ── Constants ──────────────────────────────────────────────────────────

const BINS: f32 = 256.0;
const EV_MIN: f32 = -8.0;
const EV_RANGE: f32 = 16.0;
const ZONE_COUNT: u32 = 16u;
const ZONE_BAR_H: f32 = 16.0;  // pixels in canvas space
const ZONE_GAP: f32 = 4.0;     // gap between chart and zone bar

// ── Helpers ────────────────────────────────────────────────────────────

// Sample a channel's bin with linear interpolation between adjacent bins
fn sample_channel(ch_offset: u32, bin_f: f32, max_log: f32) -> f32 {
  let b0 = u32(floor(bin_f));
  let b1 = min(b0 + 1u, 255u);
  let frac = bin_f - f32(b0);

  let v0 = f32(bins[ch_offset + b0]);
  let v1 = f32(bins[ch_offset + b1]);
  let v = mix(v0, v1, frac);

  if (v <= 0.0 || max_log <= 0.0) { return 0.0; }
  return log(1.0 + v) / max_log;
}

// Smoothstep for anti-aliased fill edge
fn fill_aa(y_norm: f32, threshold: f32, pixel_h: f32) -> f32 {
  let half_px = 0.5 / pixel_h;
  return smoothstep(threshold - half_px, threshold + half_px, y_norm);
}

// ── EV tick helpers ────────────────────────────────────────────────────

fn ev_tick_step(span: f32) -> f32 {
  if (span <= 4.0) { return 0.5; }
  if (span <= 8.0) { return 1.0; }
  if (span <= 16.0) { return 2.0; }
  return 3.0;
}

// ── Fragment ───────────────────────────────────────────────────────────

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
  let px = in.uv * vec2f(config.canvas_w, config.canvas_h);
  let lo = params.bin_lo;
  let hi = params.bin_hi;
  let max_log = params.max_log;
  let span = hi - lo;
  let is_ev = config.mode > 1.5;
  let is_luma = config.mode > 0.5 && config.mode < 1.5;
  let is_rgb = config.mode < 0.5;

  // EV mode: bottom zone_bar_h pixels are the zone bar
  let chart_h = select(config.canvas_h, config.canvas_h - ZONE_BAR_H - ZONE_GAP, is_ev);
  let in_zone_bar = is_ev && px.y > chart_h + ZONE_GAP;
  let in_chart = px.y <= chart_h;

  // Map x to fractional bin index
  let bin_f = lo + (in.uv.x * span);

  // Background color (transparent dark)
  var color = vec3f(0.0);
  var alpha = 0.0;

  // ── Zone bar (EV mode only) ────────────────────────────────────────
  if (in_zone_bar) {
    // Determine which zone this pixel is in
    let zone_f = in.uv.x * f32(ZONE_COUNT);
    let zone = u32(floor(zone_f));
    if (zone < ZONE_COUNT) {
      // Sum bins for this zone (16 bins per zone in the luma channel)
      var zone_sum = 0u;
      var max_zone = 0u;
      // We need to find max across all zones for normalization
      for (var z = 0u; z < ZONE_COUNT; z++) {
        var s = 0u;
        let z_base = z * 16u;
        for (var i = 0u; i < 16u; i++) {
          s += bins[768u + z_base + i];  // luma channel
        }
        if (z == zone) { zone_sum = s; }
        if (s > max_zone) { max_zone = s; }
      }
      if (max_zone > 0u) {
        let t = f32(zone) / f32(ZONE_COUNT - 1u);
        let brightness = (30.0 + t * 200.0) / 255.0;
        let zone_alpha = 0.15 + 0.85 * f32(zone_sum) / f32(max_zone);
        // Add 1px gap between zones
        let zone_local = fract(zone_f);
        let gap_px = 1.0 / (config.canvas_w / f32(ZONE_COUNT));
        if (zone_local > gap_px) {
          color = vec3f(brightness);
          alpha = zone_alpha;
        }
      }
    }
    return vec4f(color, alpha);
  }

  if (!in_chart) {
    return vec4f(0.0);
  }

  // Y coordinate normalized within chart area (0 = top, 1 = bottom)
  let y_norm = px.y / chart_h;

  // ── HDR overlay (rgb/luma modes: tinted region + clip line at 1.0) ──
  if (!is_ev && config.clip_bin >= lo && config.clip_bin <= hi) {
    let clip_x = (config.clip_bin - lo) / span;
    if (in.uv.x > clip_x) {
      // Tinted orange background for >1.0 region
      color = vec3f(1.0, 0.55, 0.24);
      alpha = 0.08;
    }
    // Clip line (2px wide in canvas space)
    let clip_px = clip_x * config.canvas_w;
    if (abs(px.x - clip_px) < 1.0) {
      color = vec3f(1.0, 0.55, 0.24);
      alpha = 0.4;
    }
  }

  // ── EV tick lines ──────────────────────────────────────────────────
  if (is_ev) {
    let lo_ev = EV_MIN + (lo / (BINS - 1.0)) * EV_RANGE;
    let hi_ev = EV_MIN + (hi / (BINS - 1.0)) * EV_RANGE;
    let step = ev_tick_step(hi_ev - lo_ev);
    let start = ceil(lo_ev / step) * step;

    // Iterate over possible tick positions
    var ev = start;
    for (var i = 0; i < 32; i++) {
      if (ev > hi_ev + 0.001) { break; }
      let tick_ev = round(ev / step) * step;
      let tick_bin = ((tick_ev - EV_MIN) / EV_RANGE) * (BINS - 1.0);
      let tick_x = (tick_bin - lo) / span * config.canvas_w;

      if (abs(px.x - tick_x) < 0.5) {
        let is_zero = abs(tick_ev) < 0.01;
        let tick_alpha = select(0.15, 0.35, is_zero);
        // Blend tick line over current color
        color = mix(color, vec3f(1.0), tick_alpha);
        alpha = max(alpha, tick_alpha);
      }

      ev += step;
    }
  }

  // ── Channel fills ──────────────────────────────────────────────────
  if (is_rgb) {
    // RGB mode: three channels with screen blend
    let r_h = sample_channel(0u, bin_f, max_log);
    let g_h = sample_channel(256u, bin_f, max_log);
    let b_h = sample_channel(512u, bin_f, max_log);

    let r_fill = fill_aa(y_norm, 1.0 - r_h, chart_h);
    let g_fill = fill_aa(y_norm, 1.0 - g_h, chart_h);
    let b_fill = fill_aa(y_norm, 1.0 - b_h, chart_h);

    // Channel colors with alpha
    let r_col = vec3f(0.86, 0.20, 0.20) * r_fill * 0.6;
    let g_col = vec3f(0.20, 0.78, 0.20) * g_fill * 0.6;
    let b_col = vec3f(0.20, 0.31, 0.86) * b_fill * 0.6;

    // Screen blend: 1 - (1-a)(1-b)(1-c)
    let screen = vec3f(1.0) - (vec3f(1.0) - r_col) * (vec3f(1.0) - g_col) * (vec3f(1.0) - b_col);
    let fill_alpha = max(r_fill, max(g_fill, b_fill));

    color = mix(color, screen, fill_alpha);
    alpha = max(alpha, fill_alpha * 0.6);
  } else {
    // Luma or EV mode: single channel
    let l_h = sample_channel(768u, bin_f, max_log);
    let l_fill = fill_aa(y_norm, 1.0 - l_h, chart_h);

    let fill_color = select(vec3f(0.78, 0.78, 0.78), vec3f(0.47, 0.71, 1.0), is_ev);
    let stroke_color = select(vec3f(1.0), vec3f(0.63, 0.82, 1.0), is_ev);
    let fill_alpha_val = select(0.7, 0.6, is_ev);
    let stroke_alpha_val = select(0.4, 0.5, is_ev);

    // Fill
    let ch_color = mix(fill_color, stroke_color, smoothstep(0.0, 2.0 / chart_h, y_norm - (1.0 - l_h)));
    let ch_alpha = mix(fill_alpha_val, stroke_alpha_val, smoothstep(0.0, 2.0 / chart_h, y_norm - (1.0 - l_h)));

    color = mix(color, ch_color, l_fill);
    alpha = max(alpha, l_fill * ch_alpha);
  }

  return vec4f(color, alpha);
}
