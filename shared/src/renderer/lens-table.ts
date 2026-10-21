import type { LensCurves } from '@/lib/lens/curves';

/** Radial table entries: one rgba f32 each, over ρ in [0, LENS_TABLE_DOMAIN] (ρ = 1 at the corner of the frame). */
export const LENS_TABLE_SIZE = 1024;
export const LENS_TABLE_DOMAIN = 1.25;
export const LENS_TABLE_BYTES = LENS_TABLE_SIZE * 16;

export interface LensToggles { distortion: boolean; ca: boolean; vignetting: boolean }

/** The texture the remap runs on; the optical centre defaults to its middle and the half-diagonal to half of its diagonal. */
export interface LensFrame { width: number; height: number; centre?: [number, number]; halfDiagPx?: number }

/**
 * Everything the shader needs. `table[4i..4i+3]` is (m_R, m_G, m_B, gain) at ρ_i = LENS_TABLE_DOMAIN · i / (LENS_TABLE_SIZE − 1):
 * the multipliers take an output radius to the source radius of each channel, the gain is the vignetting gain at the source radius ρ_i.
 */
export interface LensRemap { table: Float32Array; scale: number; centre: [number, number]; halfDiagPx: number; width: number; height: number }

/** Lensfun's margin on the autoscale, so the zoomed edge stays off the very last texel. */
const AUTOSCALE_MARGIN = 1.001;
const BISECTION_STEPS = 60;

const tableRadius = (i: number) => (LENS_TABLE_DOMAIN * i) / (LENS_TABLE_SIZE - 1);

/**
 * (m_R, m_G, m_B) at the output radius `rho`. m_G is the distortion; the CA curves are evaluated at the source radius m_G · rho
 * and multiply it. A correction that is off, or has no curve, contributes 1.
 */
function multipliersAt(curves: LensCurves, toggles: LensToggles, rho: number): [number, number, number] {
  const mG = toggles.distortion && curves.distortion ? curves.distortion(rho) : 1;
  const source = mG * rho;
  const mR = toggles.ca && curves.caRed ? mG * curves.caRed(source) : mG;
  const mB = toggles.ca && curves.caBlue ? mG * curves.caBlue(source) : mG;
  return [mR, mG, mB];
}

/** The output radius, in ρ, at which the farthest-reaching channel samples the boundary at distance `d`; null when the domain falls short of it. */
function boundaryRadius(curves: LensCurves, toggles: LensToggles, d: number): number | null {
  const f = (rho: number) => Math.max(...multipliersAt(curves, toggles, rho)) * rho - d;
  if (!(f(LENS_TABLE_DOMAIN) >= 0)) return null;
  let lo = 0, hi = LENS_TABLE_DOMAIN;
  for (let i = 0; i < BISECTION_STEPS; i++) {
    const mid = (lo + hi) / 2;
    if (f(mid) >= 0) hi = mid; else lo = mid;
  }
  return hi;
}

/**
 * The zoom that keeps every channel's source inside the frame (lensfun's autoscale): below 1 for barrel, above 1 for pincushion.
 * It is the largest ratio of boundary distance to solved radius over the corners (distance 1) and the edge midpoints,
 * which are symmetric, so three distances stand for lensfun's eight directions. 1 when the table's domain cannot reach a boundary.
 */
export function autoscale(curves: LensCurves, toggles: LensToggles, frame: LensFrame): number {
  const diagonal = Math.hypot(frame.width, frame.height);
  let scale = 0;
  for (const d of [1, frame.width / diagonal, frame.height / diagonal]) {
    const rho = boundaryRadius(curves, toggles, d);
    if (rho === null) return 1;
    scale = Math.max(scale, d / rho);
  }
  return scale * AUTOSCALE_MARGIN;
}

/** The radial table of the enabled corrections and the autoscale; null when none of them has a curve. */
export function buildLensTable(curves: LensCurves, toggles: LensToggles, frame: LensFrame): LensRemap | null {
  const vignetting = toggles.vignetting ? curves.vignetting : null;
  const corrects = (toggles.distortion && curves.distortion) || (toggles.ca && (curves.caRed || curves.caBlue)) || vignetting;
  if (!corrects) return null;

  const table = new Float32Array(LENS_TABLE_SIZE * 4);
  for (let i = 0; i < LENS_TABLE_SIZE; i++) {
    const rho = tableRadius(i);
    table.set([...multipliersAt(curves, toggles, rho), vignetting ? vignetting(rho) : 1], i * 4);
  }
  const { width, height } = frame;
  return {
    table,
    scale: autoscale(curves, toggles, frame),
    centre: frame.centre ? [frame.centre[0], frame.centre[1]] : [width / 2, height / 2],
    halfDiagPx: frame.halfDiagPx ?? Math.hypot(width, height) / 2,
    width,
    height,
  };
}

/** Linear interpolation of one table channel at the radius `rho`, held at the ends. */
function lookup(table: Float32Array, rho: number, channel: number): number {
  const x = Math.min(Math.max((rho / LENS_TABLE_DOMAIN) * (LENS_TABLE_SIZE - 1), 0), LENS_TABLE_SIZE - 1);
  const i = Math.min(Math.floor(x), LENS_TABLE_SIZE - 2);
  const a = table[i * 4 + channel];
  return a + (table[(i + 1) * 4 + channel] - a) * (x - i);
}

/**
 * The CPU twin of the shader's remap: where each channel of the output pixel at `uv` samples the source, and the vignetting gain.
 * The offset from the optical centre is scaled down by the autoscale before the table is read.
 */
export function lensSourceUv(remap: LensRemap, uv: [number, number]): { r: [number, number]; g: [number, number]; b: [number, number]; gain: number } {
  const { table, scale, centre, halfDiagPx, width, height } = remap;
  const dx = ((uv[0] - 0.5) * width - (centre[0] - width / 2)) / scale;
  const dy = ((uv[1] - 0.5) * height - (centre[1] - height / 2)) / scale;
  const rho = Math.hypot(dx, dy) / halfDiagPx;
  const mR = lookup(table, rho, 0), mG = lookup(table, rho, 1), mB = lookup(table, rho, 2);
  const sourceUv = (m: number): [number, number] => [(centre[0] + dx * m) / width, (centre[1] + dy * m) / height];
  return { r: sourceUv(mR), g: sourceUv(mG), b: sourceUv(mB), gain: lookup(table, rho * mG, 3) };
}
