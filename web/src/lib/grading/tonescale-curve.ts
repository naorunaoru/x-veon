// Models the CORE OpenDRT tonescale (Contrast / Shoulder / Toe) from
// gl/shaders/opendrt.wgsl, for a read-only Tone Curve visualization. It is
// bit-exact for tn_con / tn_sh / tn_toe — the controls this panel exposes.
// It intentionally omits the pre-tonescale local-contrast (tn_lcon), offset
// (tn_off), and high-contrast (tn_hcon) steps and the purity/hue paths, so on
// looks that enable lcon (e.g. 'default') the plotted curve is a close
// approximation (within a few % code value), not the full achromatic transfer.
import type { GradingConfig, TonescaleParams } from '@/gl/opendrt-params';

function spowf(x: number, p: number): number {
  return x <= 0 ? x : Math.pow(x, p);
}
function compressHp(x: number, s: number, p: number): number {
  return spowf(x / (x + s), p);
}
function compressToeQuad(x: number, toe: number): number {
  return toe === 0 ? x : spowf(x, 2) / (x + toe);
}
function srgbOetf(v: number): number {
  const c = Math.min(1, Math.max(0, v));
  return c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
}

/** Display code value (0..1) for a scene-linear achromatic input. */
export function evalTonescale(sceneLinear: number, cfg: GradingConfig, ts: TonescaleParams): number {
  let v = compressHp(sceneLinear, ts.ts_s, cfg.tn_con);
  v = v * ts.ts_m2;                       // normalization (shader opendrt.wgsl:439)
  v = compressToeQuad(v, cfg.tn_toe);     // toe (opendrt.wgsl:440)
  v = v * ts.ts_dsc;                      // display scale (opendrt.wgsl:441)
  return srgbOetf(v);
}

export interface CurvePoint { x: number; y: number; }

/**
 * n points over a fixed scene exposure range (−8…+6 stops around middle grey),
 * normalized to [0,1] for plotting. x is the normalized log-scene axis, y the
 * display code value.
 */
export function sampleToneCurve(cfg: GradingConfig, ts: TonescaleParams, n: number): CurvePoint[] {
  const lo = -8, hi = 6;
  const pts: CurvePoint[] = [];
  for (let i = 0; i < n; i++) {
    const t = n === 1 ? 0 : i / (n - 1);
    const scene = 0.18 * Math.pow(2, lo + t * (hi - lo));
    pts.push({ x: t, y: evalTonescale(scene, cfg, ts) });
  }
  return pts;
}
