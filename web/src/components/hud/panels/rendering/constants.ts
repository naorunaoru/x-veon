// Static data for the Rendering panel: hue geometry + per-mode key maps,
// wheel modes and the Advanced drawer schema.
import { OPENDRT_LIMITS, type OpenDrtConfig } from '@/renderer/grading/opendrt-params';

type DrtKey = keyof OpenDrtConfig;
export type WheelMode = 'brl' | 'hue' | 'pur';

/** Hex node colours — final, from the design tokens (channel-ish but tuned for the wheel). */
export const HUE_COLORS = {
  R: '#E0564E', Y: '#D8B13A', G: '#5BB85F', C: '#3FB6C4', B: '#5B82E0', M: '#C254C0',
} as const;
export type HueKey = keyof typeof HUE_COLORS;

export interface Hue {
  k: HueKey;
  deg: number;
  col: string;
  /** Real OpenDRT key edited per wheel mode. `pur` only has primaries (R/G/B). */
  keys: { brl: DrtKey; hue: DrtKey; pur: DrtKey | null };
}

// Order matters — the per-hue chip row and node iteration follow this list.
export const HUES: Hue[] = [
  { k: 'R', deg: 0,   col: HUE_COLORS.R, keys: { brl: 'brl_r', hue: 'hs_r', pur: 'pt_r' } },
  { k: 'Y', deg: 60,  col: HUE_COLORS.Y, keys: { brl: 'brl_y', hue: 'hs_y', pur: null } },
  { k: 'G', deg: 120, col: HUE_COLORS.G, keys: { brl: 'brl_g', hue: 'hs_g', pur: 'pt_g' } },
  { k: 'C', deg: 180, col: HUE_COLORS.C, keys: { brl: 'brl_c', hue: 'hs_c', pur: null } },
  { k: 'B', deg: 240, col: HUE_COLORS.B, keys: { brl: 'brl_b', hue: 'hs_b', pur: 'pt_b' } },
  { k: 'M', deg: 300, col: HUE_COLORS.M, keys: { brl: 'brl_m', hue: 'hs_m', pur: null } },
];

export const WHEEL_MODES: { k: WheelMode; label: string }[] = [
  { k: 'brl', label: 'Brilliance' },
  { k: 'hue', label: 'Hue twist' },
  { k: 'pur', label: 'Purity' },
];

/** The OpenDRT key the wheel's "Intensity range" slider drives, per mode. */
export const RANGE_KEY: Record<WheelMode, { key: DrtKey; min: number; max: number }> = {
  brl: { key: 'brl_rng', min: 0, max: 1 },
  hue: { key: 'hs_rgb_rng', min: OPENDRT_LIMITS.hs_rgb_rng[0], max: OPENDRT_LIMITS.hs_rgb_rng[1] },
  pur: { key: 'pt_rng_low', min: OPENDRT_LIMITS.pt_rng_low[0], max: OPENDRT_LIMITS.pt_rng_low[1] },
};

// ── Purity (pt_r/g/b) ↔ wheel transform ────────────────────────────────────
// pt_* are compression amounts (0..5, higher = MORE compression = LESS purity).
// The wheel node reads as a signed deviation from the look's value, in [-1,1],
// inverted so "outward = +saturation": v>0 lowers pt toward 0 (max purity),
// v<0 raises pt toward 5 (max compression). v=0 sits on the base ring = look value.
export const PT_MAX = 5;
export function ptToWheel(pt: number, base: number): number {
  if (pt <= base) return base === 0 ? 0 : (base - pt) / base;          // 0..+1
  return -(pt - base) / (PT_MAX - base || 1);                          // -1..0
}
export function wheelToPt(v: number, base: number): number {
  const pt = v >= 0 ? base * (1 - v) : base + (PT_MAX - base) * -v;
  return +pt.toFixed(3);
}

// ── Expert / stickshift schema ──────────────────────────────────────────────
// Every underlying OpenDRT knob, by module. Generated coverage of the full
// OpenDrtConfig so no parameter is unreachable from the UI.
export type ExpertRow =
  | { kind: 'slider'; key: DrtKey; label: string; min: number; max: number; step: number }
  | { kind: 'toggle'; key: DrtKey; label: string };

const s = (key: DrtKey, label: string, min: number, max: number, step: number): ExpertRow =>
  ({ kind: 'slider', key, label,
    min: key in OPENDRT_LIMITS ? OPENDRT_LIMITS[key as keyof typeof OPENDRT_LIMITS][0] : min,
    max: key in OPENDRT_LIMITS ? Math.min(max, OPENDRT_LIMITS[key as keyof typeof OPENDRT_LIMITS][1]) : max, step });
const t = (key: DrtKey, label: string): ExpertRow => ({ kind: 'toggle', key, label });

export const EXPERT_GROUPS: { title: string; rows: ExpertRow[] }[] = [
  { title: 'Tonescale', rows: [
    s('tn_lg', 'Middle grey', 3, 25, 0.1),
    s('tn_con', 'Contrast', 0.9, 2.1, 0.01),
    s('tn_sh', 'Shoulder', 0, 1, 0.01),
    s('tn_toe', 'Toe', 0, 0.1, 0.001),
    s('tn_off', 'Offset', 0, 0.05, 0.001),
  ] },
  { title: 'Low contrast', rows: [
    t('tn_lcon_enable', 'Enable'),
    s('tn_lcon', 'Amount', 0, 2, 0.01),
    s('tn_lcon_w', 'Width', 0, 2, 0.01),
    s('tn_lcon_pc', 'Per-channel', 0, 1, 0.01),
  ] },
  { title: 'High contrast', rows: [
    t('tn_hcon_enable', 'Enable'),
    s('tn_hcon', 'Amount', -1, 1, 0.01),
    s('tn_hcon_pv', 'Pivot', 0, 4, 0.01),
    s('tn_hcon_st', 'Strength', 0, 8, 0.01),
  ] },
  { title: 'Purity', rows: [
    s('rs_sa', 'Render strength', 0, 1, 0.01),
    s('rs_rw', 'Render red wt', 0, 1, 0.01),
    s('rs_bw', 'Render blue wt', 0, 1, 0.01),
    s('pt_r', 'Compress R', 0, 5, 0.01),
    s('pt_g', 'Compress G', 0, 5, 0.01),
    s('pt_b', 'Compress B', 0, 5, 0.01),
    s('pt_rng_low', 'Range low', 0, 1, 0.01),
    s('pt_rng_high', 'Range high', 0.25, 2, 0.01),
    t('ptl_enable', 'Compress low'),
    t('ptm_enable', 'Mid purity'),
    s('ptm_low', 'Mid low', -1, 1, 0.01),
    s('ptm_low_st', 'Mid low strength', 0, 1, 0.01),
    s('ptm_high', 'Mid high', -1, 1, 0.01),
    s('ptm_high_st', 'Mid high strength', 0, 1, 0.01),
  ] },
  { title: 'Brilliance', rows: [
    t('brl_enable', 'Enable'),
    s('brl_r', 'Red', -1, 1, 0.01),
    s('brl_g', 'Green', -1, 1, 0.01),
    s('brl_b', 'Blue', -1, 1, 0.01),
    s('brl_c', 'Cyan', -1, 1, 0.01),
    s('brl_m', 'Magenta', -1, 1, 0.01),
    s('brl_y', 'Yellow', -1, 1, 0.01),
    s('brl_rng', 'Range', 0, 1, 0.01),
  ] },
  { title: 'Hue', rows: [
    t('hs_rgb_enable', 'RGB enable'),
    s('hs_r', 'Shift R', -1, 2, 0.01),
    s('hs_g', 'Shift G', -1, 2, 0.01),
    s('hs_b', 'Shift B', -1, 2, 0.01),
    s('hs_rgb_rng', 'RGB range', 0, 4, 0.01),
    t('hs_cmy_enable', 'CMY enable'),
    s('hs_c', 'Shift C', -1, 2, 0.01),
    s('hs_m', 'Shift M', -1, 2, 0.01),
    s('hs_y', 'Shift Y', -1, 2, 0.01),
    t('hc_enable', 'Hue contrast'),
    s('hc_r', 'Hue contrast R', 0, 2, 0.01),
  ] },
  { title: 'Whites', rows: [
    s('cwp', 'Creative white', 0, 1, 0.01),
    s('cwp_rng', 'White range', 0, 1, 0.01),
  ] },
  { title: 'Output', rows: [
    s('peak_luminance', 'Peak luminance', 100, 2000, 10),
    s('grey_boost', 'Grey boost', 0, 0.5, 0.01),
    s('pt_hdr', 'HDR purity blend', 0, 1, 0.01),
  ] },
];
