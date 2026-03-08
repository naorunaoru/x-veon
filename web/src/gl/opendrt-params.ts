// TypeScript port of OpenDrtConfig presets and TonescaleParams from opendrt.rs.
// Only config values + precomputed tonescale constants — process_pixel runs in the shader.

import type { LookPreset } from '@/pipeline/types';

export interface OpenDrtConfig {
  tn_lg: number;
  tn_con: number;
  tn_sh: number;
  tn_toe: number;
  tn_off: number;
  tn_lcon_enable: boolean;
  tn_lcon: number;
  tn_lcon_w: number;
  tn_lcon_pc: number;
  tn_hcon_enable: boolean;
  tn_hcon: number;
  tn_hcon_pv: number;
  tn_hcon_st: number;
  rs_sa: number;
  rs_rw: number;
  rs_bw: number;
  pt_r: number;
  pt_g: number;
  pt_b: number;
  pt_rng_low: number;
  pt_rng_high: number;
  ptl_enable: boolean;
  ptm_enable: boolean;
  ptm_low: number;
  ptm_low_st: number;
  ptm_high: number;
  ptm_high_st: number;
  brl_enable: boolean;
  brl_r: number;
  brl_g: number;
  brl_b: number;
  brl_c: number;
  brl_m: number;
  brl_y: number;
  brl_rng: number;
  hs_rgb_enable: boolean;
  hs_r: number;
  hs_g: number;
  hs_b: number;
  hs_rgb_rng: number;
  hs_cmy_enable: boolean;
  hs_c: number;
  hs_m: number;
  hs_y: number;
  hc_enable: boolean;
  hc_r: number;
  cwp: number;      // 0 = D65 (off), 1 = D50 (full warm)
  cwp_rng: number;  // range of creative white blend (0-1)
  peak_luminance: number;
  grey_boost: number;
  pt_hdr: number;
}

/** Pre-processing adjustments applied before OpenDRT in the shader. */
export interface PreProcessConfig {
  exposure: number;   // EV stops
  wb_temp: number;    // white balance temperature correction: warm(+) / cool(-)
  wb_tint: number;    // white balance tint correction: magenta(+) / green(-)
  sharpen_amount: number; // unsharp mask strength (0 = off)
}

/** Full grading config: OpenDRT pipeline + pre-processing. */
export type GradingConfig = OpenDrtConfig & PreProcessConfig;

export interface TonescaleParams {
  ts_s: number;
  ts_s1: number;
  ts_m2: number;
  ts_dsc: number;
  ts_x0: number;
}

// ── Math helpers (matching Rust exactly) ──────────────────────────────────

function spowf(x: number, p: number): number {
  return x <= 0 ? x : Math.pow(x, p);
}

function compress_toe_quadratic_inv(x: number, toe: number): number {
  if (toe === 0) return x;
  return (x + Math.sqrt(x * (4 * toe + x))) / 2;
}

function compress_hp(x: number, s: number, p: number): number {
  return spowf(x / (x + s), p);
}

// ── Presets (matching opendrt.rs lines 96-197) ────────────────────────────

export const DEFAULT_PREPROCESS: PreProcessConfig = {
  exposure: 0.0,
  wb_temp: 0.0,
  wb_tint: 0.0,
  sharpen_amount: 0.0,
};

function baseSdr(): OpenDrtConfig {
  return {
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
    hs_rgb_enable: false,
    hs_r: 0.0,
    hs_g: 0.0,
    hs_b: 0.0,
    hs_rgb_rng: 1.0,
    hs_cmy_enable: false,
    hs_c: 0.0,
    hs_m: 0.0,
    hs_y: 0.0,
    hc_enable: false,
    hc_r: 0.0,
    cwp: 0.0,
    cwp_rng: 0.0,
    peak_luminance: 100.0,
    grey_boost: 0.13,
    pt_hdr: 0.5,
  };
}

function defaultSdr(): OpenDrtConfig {
  return {
    ...baseSdr(),
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
    hs_rgb_enable: true,
    hs_r: 0.6,
    hs_g: 0.35,
    hs_b: 0.66,
    hs_rgb_rng: 1.0,
    hs_cmy_enable: true,
    hs_c: 0.25,
    hs_m: 0.0,
    hs_y: 0.0,
    hc_enable: true,
    hc_r: 1.0,
  };
}

function colorfulSdr(): OpenDrtConfig {
  return {
    ...baseSdr(),
    tn_con: 1.5,
    tn_sh: 0.5,
    tn_toe: 0.003,
    tn_off: 0.003,
    tn_lcon_enable: true,
    tn_lcon: 0.4,
    tn_lcon_w: 0.5,
    rs_bw: 0.55,
    pt_r: 0.5,
    pt_g: 2.0,
    pt_b: 2.0,
    pt_rng_low: 0.2,
    pt_rng_high: 0.8,
    ptm_enable: true,
    ptm_low: 0.4,
    ptm_low_st: 0.5,
    ptm_high: -0.6,
    ptm_high_st: 0.3,
    brl_enable: true,
    brl_r: -0.5,
    brl_g: -0.4,
    brl_b: -0.2,
    brl_c: 0.0,
    brl_m: 0.0,
    brl_y: 0.0,
    brl_rng: 0.66,
    hs_rgb_enable: true,
    hs_r: 0.5,
    hs_g: 0.35,
    hs_b: 0.5,
    hs_rgb_rng: 1.0,
    hs_cmy_enable: true,
    hs_c: 0.25,
    hs_m: 0.0,
    hs_y: 0.25,
    hc_enable: true,
    hc_r: 1.0,
  };
}

function umbraSdr(): OpenDrtConfig {
  return {
    ...baseSdr(),
    tn_con: 1.8,
    tn_sh: 0.5,
    tn_toe: 0.001,
    tn_off: 0.015,
    tn_lcon_enable: true,
    tn_lcon: 1.0,
    tn_lcon_w: 1.0,
    rs_bw: 0.55,
    pt_r: 0.5,
    pt_g: 2.0,
    pt_b: 2.5,
    pt_rng_low: 0.25,
    pt_rng_high: 0.25,
    ptl_enable: true,
    ptm_enable: true,
    ptm_low: 0.4,
    ptm_low_st: 0.66,
    ptm_high: -0.6,
    ptm_high_st: 0.45,
    brl_enable: true,
    brl_r: -0.5,
    brl_g: -0.4,
    brl_b: -0.2,
    brl_c: 0.0,
    brl_m: 0.0,
    brl_y: 0.0,
    brl_rng: 0.35,
    hs_rgb_enable: true,
    hs_r: 0.66,
    hs_g: 0.5,
    hs_b: 0.85,
    hs_rgb_rng: 2.0,
    hs_cmy_enable: true,
    hs_c: 0.0,
    hs_m: 0.25,
    hs_y: 0.66,
    hc_enable: true,
    hc_r: 1.0,
    cwp: 1.0,
    cwp_rng: 0.25,
  };
}

function flatSdr(): OpenDrtConfig {
  return {
    ...baseSdr(),
    tn_con: 1.15,
    rs_sa: 0.2,
  };
}

// ── Tonescale Presets ──────────────────────────────────────────────────────

export type TonescalePreset = 'low-contrast' | 'medium-contrast' | 'high-contrast' |
  'arriba' | 'sylvan' | 'colorful' | 'aery' | 'dystopic' | 'umbra' |
  'aces-1x' | 'aces-2' | 'marvelous' | 'dagrinchi';

type TonescaleOverrides = Pick<OpenDrtConfig,
  'tn_con' | 'tn_sh' | 'tn_toe' | 'tn_off' |
  'tn_hcon_enable' | 'tn_hcon' | 'tn_hcon_pv' | 'tn_hcon_st' |
  'tn_lcon_enable' | 'tn_lcon' | 'tn_lcon_w'>;

export const TONESCALE_PRESETS: Record<TonescalePreset, { label: string; overrides: TonescaleOverrides }> = {
  'low-contrast': {
    label: 'Low Contrast',
    overrides: {
      tn_con: 1.4, tn_sh: 0.5, tn_toe: 0.003, tn_off: 0.005,
      tn_hcon_enable: false, tn_hcon: 0.0, tn_hcon_pv: 1.0, tn_hcon_st: 4.0,
      tn_lcon_enable: false, tn_lcon: 0.0, tn_lcon_w: 0.5,
    },
  },
  'medium-contrast': {
    label: 'Medium Contrast',
    overrides: {
      tn_con: 1.66, tn_sh: 0.5, tn_toe: 0.003, tn_off: 0.005,
      tn_hcon_enable: false, tn_hcon: 0.0, tn_hcon_pv: 1.0, tn_hcon_st: 4.0,
      tn_lcon_enable: false, tn_lcon: 0.0, tn_lcon_w: 0.5,
    },
  },
  'high-contrast': {
    label: 'High Contrast',
    overrides: {
      tn_con: 1.4, tn_sh: 0.5, tn_toe: 0.003, tn_off: 0.005,
      tn_hcon_enable: false, tn_hcon: 0.0, tn_hcon_pv: 1.0, tn_hcon_st: 4.0,
      tn_lcon_enable: true, tn_lcon: 1.0, tn_lcon_w: 0.5,
    },
  },
  'arriba': {
    label: 'Arriba',
    overrides: {
      tn_con: 1.05, tn_sh: 0.5, tn_toe: 0.1, tn_off: 0.01,
      tn_hcon_enable: false, tn_hcon: 0.0, tn_hcon_pv: 1.0, tn_hcon_st: 4.0,
      tn_lcon_enable: true, tn_lcon: 1.5, tn_lcon_w: 0.2,
    },
  },
  'sylvan': {
    label: 'Sylvan',
    overrides: {
      tn_con: 1.6, tn_sh: 0.5, tn_toe: 0.01, tn_off: 0.01,
      tn_hcon_enable: false, tn_hcon: 0.0, tn_hcon_pv: 1.0, tn_hcon_st: 4.0,
      tn_lcon_enable: true, tn_lcon: 0.25, tn_lcon_w: 0.75,
    },
  },
  'colorful': {
    label: 'Colorful',
    overrides: {
      tn_con: 1.5, tn_sh: 0.5, tn_toe: 0.003, tn_off: 0.003,
      tn_hcon_enable: false, tn_hcon: 0.0, tn_hcon_pv: 1.0, tn_hcon_st: 4.0,
      tn_lcon_enable: true, tn_lcon: 0.4, tn_lcon_w: 0.5,
    },
  },
  'aery': {
    label: 'Aery',
    overrides: {
      tn_con: 1.15, tn_sh: 0.5, tn_toe: 0.04, tn_off: 0.006,
      tn_hcon_enable: false, tn_hcon: 0.0, tn_hcon_pv: 0.0, tn_hcon_st: 0.5,
      tn_lcon_enable: true, tn_lcon: 0.5, tn_lcon_w: 2.0,
    },
  },
  'dystopic': {
    label: 'Dystopic',
    overrides: {
      tn_con: 1.6, tn_sh: 0.5, tn_toe: 0.01, tn_off: 0.008,
      tn_hcon_enable: true, tn_hcon: 0.25, tn_hcon_pv: 0.0, tn_hcon_st: 1.0,
      tn_lcon_enable: true, tn_lcon: 1.0, tn_lcon_w: 0.75,
    },
  },
  'umbra': {
    label: 'Umbra',
    overrides: {
      tn_con: 1.8, tn_sh: 0.5, tn_toe: 0.001, tn_off: 0.015,
      tn_hcon_enable: false, tn_hcon: 0.0, tn_hcon_pv: 1.0, tn_hcon_st: 4.0,
      tn_lcon_enable: true, tn_lcon: 1.0, tn_lcon_w: 1.0,
    },
  },
  'aces-1x': {
    label: 'ACES 1.x',
    overrides: {
      tn_con: 1.0, tn_sh: 0.35, tn_toe: 0.02, tn_off: 0.0,
      tn_hcon_enable: true, tn_hcon: 0.55, tn_hcon_pv: 0.0, tn_hcon_st: 2.0,
      tn_lcon_enable: true, tn_lcon: 1.13, tn_lcon_w: 1.0,
    },
  },
  'aces-2': {
    label: 'ACES 2.0',
    overrides: {
      tn_con: 1.15, tn_sh: 0.5, tn_toe: 0.04, tn_off: 0.0,
      tn_hcon_enable: false, tn_hcon: 1.0, tn_hcon_pv: 1.0, tn_hcon_st: 1.0,
      tn_lcon_enable: false, tn_lcon: 1.0, tn_lcon_w: 0.6,
    },
  },
  'marvelous': {
    label: 'Marvelous',
    overrides: {
      tn_con: 1.5, tn_sh: 0.5, tn_toe: 0.003, tn_off: 0.01,
      tn_hcon_enable: true, tn_hcon: 0.25, tn_hcon_pv: 0.0, tn_hcon_st: 4.0,
      tn_lcon_enable: true, tn_lcon: 1.0, tn_lcon_w: 1.0,
    },
  },
  'dagrinchi': {
    label: 'DaGrinchi',
    overrides: {
      tn_con: 1.2, tn_sh: 0.5, tn_toe: 0.02, tn_off: 0.0,
      tn_hcon_enable: false, tn_hcon: 0.0, tn_hcon_pv: 1.0, tn_hcon_st: 1.0,
      tn_lcon_enable: false, tn_lcon: 0.0, tn_lcon_w: 0.6,
    },
  },
};

export function configFromPreset(preset: LookPreset, hdrHeadroom?: number): OpenDrtConfig {
  const presetMap: Record<LookPreset, () => OpenDrtConfig> = {
    default: defaultSdr,
    colorful: colorfulSdr,
    umbra: umbraSdr,
    base: baseSdr,
    flat: flatSdr,
  };
  const cfg = (presetMap[preset] ?? defaultSdr)();
  if (hdrHeadroom != null && hdrHeadroom > 1.0) {
    cfg.peak_luminance = hdrHeadroom * 100;
  }
  return cfg;
}

/** Merge user overrides on top of a preset config. Auto-enables feature groups when relevant params are overridden. */
export function configWithOverrides(
  base: OpenDrtConfig,
  overrides: Partial<OpenDrtConfig>,
  preProcess?: Partial<PreProcessConfig>,
): GradingConfig {
  const cfg: GradingConfig = { ...DEFAULT_PREPROCESS, ...base, ...overrides, ...preProcess };
  // Auto-enable feature groups when their params are explicitly set
  if ('tn_lcon' in overrides && !('tn_lcon_enable' in overrides)) {
    cfg.tn_lcon_enable = cfg.tn_lcon !== 0;
  }
  if (('brl_r' in overrides || 'brl_g' in overrides || 'brl_b' in overrides ||
       'brl_c' in overrides || 'brl_m' in overrides || 'brl_y' in overrides) &&
      !('brl_enable' in overrides)) {
    cfg.brl_enable = true;
  }
  if (('hs_r' in overrides || 'hs_g' in overrides || 'hs_b' in overrides) &&
      !('hs_rgb_enable' in overrides)) {
    cfg.hs_rgb_enable = true;
  }
  if (('hs_c' in overrides || 'hs_m' in overrides || 'hs_y' in overrides) &&
      !('hs_cmy_enable' in overrides)) {
    cfg.hs_cmy_enable = true;
  }
  if ('hc_r' in overrides && !('hc_enable' in overrides)) {
    cfg.hc_enable = cfg.hc_r !== 0;
  }
  return cfg;
}

/** Derive an HDR variant from an SDR config (for JPEG-HDR / AVIF export). */
export function deriveHdrConfig(sdrConfig: GradingConfig, peakLuminance = 1000): GradingConfig {
  return { ...sdrConfig, peak_luminance: peakLuminance };
}

// ── TonescaleParams (matching opendrt.rs TonescaleParams::new, lines 209-230) ──

export function computeTonescaleParams(cfg: OpenDrtConfig): TonescaleParams {
  const ts_x1 = Math.pow(2, 6.0 * cfg.tn_sh + 4.0);
  const ts_y1 = cfg.peak_luminance / 100.0;
  const ts_x0 = 0.18 + cfg.tn_off;
  const ts_y0 = (cfg.tn_lg / 100.0) * (1.0 + cfg.grey_boost * Math.log2(Math.max(ts_y1, 1e-10)));
  const ts_s0 = compress_toe_quadratic_inv(ts_y0, cfg.tn_toe);
  const ts_s10 = ts_x0 * (spowf(ts_s0, -1.0 / cfg.tn_con) - 1.0);
  const ts_m1 = ts_y1 / spowf(ts_x1 / (ts_x1 + ts_s10), cfg.tn_con);
  const ts_m2 = compress_toe_quadratic_inv(ts_m1, cfg.tn_toe);
  const ts_s = ts_x0 * (spowf(ts_s0 / ts_m2, -1.0 / cfg.tn_con) - 1.0);
  const ts_dsc = 100.0 / cfg.peak_luminance;

  // HDR purity blend
  const pt_cmp_lf = cfg.pt_hdr * Math.min((cfg.peak_luminance - 100.0) / 900.0, 1.0);
  const s_lp100 = ts_x0 * (spowf(cfg.tn_lg / 100.0, -1.0 / cfg.tn_con) - 1.0);
  const ts_s1 = ts_s * pt_cmp_lf + s_lp100 * (1.0 - pt_cmp_lf);

  return { ts_s, ts_s1, ts_m2, ts_dsc, ts_x0 };
}
