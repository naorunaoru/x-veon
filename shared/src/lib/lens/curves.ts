import type { LensProfile, LfDistortion, LfTCA, LfVignetting } from '../lensfun-types';
import type { LensManual } from './edit';

/**
 * Radial correction curves, in units of the half-diagonal of the frame (ρ = 1 at the corner).
 * A null curve is a correction that is absent.
 */
export interface LensCurves {
  distortion: ((rhoOut: number) => number) | null; // m_G: source radius / output radius, at the output radius
  caRed: ((rhoSrc: number) => number) | null;       // extra factor on the source radius, for red
  caBlue: ((rhoSrc: number) => number) | null;
  vignetting: ((rhoSrc: number) => number) | null;  // gain at the source radius
}

export const NO_CURVES: LensCurves = Object.freeze({ distortion: null, caRed: null, caBlue: null, vignetting: null });

/** Piecewise-linear interpolation through (xs[i], ys[i]); `xs` ascending. Holds the end values outside. */
export function linearSpline(xs: number[], ys: number[]): (x: number) => number {
  const last = xs.length - 1;
  return (x) => {
    if (x <= xs[0]) return ys[0];
    if (x >= xs[last]) return ys[last];
    let i = 1;
    while (xs[i] < x) i++;
    const t = (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
    return ys[i - 1] + t * (ys[i] - ys[i - 1]);
  };
}

// ── lensfun model evaluation (hugin units, formulas of lensfun.h.in) ──

/** r_d / r_u at the undistorted radius `ru` (hugin units): the model's polynomial, so its constant term at the centre. An unknown model is 1. */
function distortionRatio(entry: LfDistortion, ru: number): number {
  const r2 = ru * ru;
  switch (entry.model) {
    case 'poly3': {
      const k1 = entry.k1 ?? 0;
      return 1 - k1 + k1 * r2;
    }
    case 'poly5':
      return 1 + (entry.k1 ?? 0) * r2 + (entry.k2 ?? 0) * r2 * r2;
    case 'ptlens': {
      const a = entry.a ?? 0, b = entry.b ?? 0, c = entry.c ?? 0;
      return a * ru * r2 + b * r2 + c * ru + 1 - a - b - c;
    }
    default:
      return 1;
  }
}

/** Per-channel r_d / r at the radius `r` that distortion already produced (hugin units). An unknown model is 1. */
function tcaRatio(entry: LfTCA, r: number): { red: number; blue: number } {
  switch (entry.model) {
    case 'linear':
      return { red: entry.kr ?? 1, blue: entry.kb ?? 1 };
    case 'poly3':
      return {
        red: (entry.br ?? 0) * r * r + (entry.cr ?? 0) * r + (entry.vr ?? 1),
        blue: (entry.bb ?? 0) * r * r + (entry.cb ?? 0) * r + (entry.vb ?? 1),
      };
    default:
      return { red: 1, blue: 1 };
  }
}

/** Distorted (source) radius r_d of an undistorted (output) radius `ru`, both in hugin units. An unknown model is the identity. */
export function evalDistortion(entry: LfDistortion, ru: number): number {
  return ru * distortionRatio(entry, ru);
}

/** Per-channel source radius for a radius `r` that distortion already produced, in hugin units. An unknown model is the identity. */
export function evalTca(entry: LfTCA, r: number): { red: number; blue: number } {
  const m = tcaRatio(entry, r);
  return { red: m.red * r, blue: m.blue * r };
}

/** The smallest vignetting denominator used: caps the gain at 20 (about 4.3 stops). */
const MIN_VIGNETTING_DENOMINATOR = 0.05;

/**
 * Brightness gain at source radius `r` (half-diagonal units of the calibration sensor). An unknown model is 1.
 * A lens calibrated on a smaller sensor than the camera's reaches r > 1, outside its calibration, where the polynomial
 * can cross zero: the denominator is held above a floor so the gain stays finite and positive.
 */
export function evalVignettingGain(entry: LfVignetting, r: number): number {
  if (entry.model !== 'pa') return 1;
  const r2 = r * r;
  return 1 / Math.max(1 + entry.k1 * r2 + entry.k2 * r2 * r2 + entry.k3 * r2 * r2 * r2, MIN_VIGNETTING_DENOMINATOR);
}

// ── Units ───────────────────────────────────────────────────────────

/**
 * Multipliers from our radius (ρ, half-diagonal of the visible image) to lensfun's: `geometry` for distortion and
 * TCA (hugin units, 1 at half the calibration sensor's shorter side), `vignetting` for vignetting (1 at its half-diagonal).
 */
export function huginUnits(p: Pick<LensProfile, 'cropfactor' | 'calibCropfactor' | 'calibAspectRatio'>): { geometry: number; vignetting: number } {
  const scale = p.calibCropfactor / p.cropfactor;
  return { geometry: scale * Math.hypot(p.calibAspectRatio, 1), vignetting: scale };
}

// ── lensfun selection and blending ──────────────────────────────────

type Curve = (x: number) => number;

/**
 * The curves of the two entries bracketing `x` along `pos`, blended linearly in `pos`; the nearest entry's curve
 * outside the range. `entries` is not empty.
 */
function blendEntries<E>(entries: E[], pos: (e: E) => number, x: number, curveOf: (e: E) => Curve): Curve {
  const sorted = [...entries].sort((p, q) => pos(p) - pos(q));
  const last = sorted.length - 1;
  if (x <= pos(sorted[0])) return curveOf(sorted[0]);
  if (x >= pos(sorted[last])) return curveOf(sorted[last]);
  let i = 1;
  while (pos(sorted[i]) < x) i++;
  const w = (x - pos(sorted[i - 1])) / (pos(sorted[i]) - pos(sorted[i - 1]));
  const a = curveOf(sorted[i - 1]);
  const b = curveOf(sorted[i]);
  return (r) => (1 - w) * a(r) + w * b(r);
}

/** A focal length or f-number from EXIF; 0, negative or non-finite means it is not known. */
const isKnown = (x: number) => Number.isFinite(x) && x > 0;

/** Distortion and TCA entries: bracketed by focal length. An unknown focal length takes the only entry, if there is exactly one. */
function byFocal<E extends { focal: number }>(entries: E[], focal: number, curveOf: (e: E) => Curve): Curve | null {
  if (entries.length === 0) return null;
  if (!isKnown(focal)) return entries.length === 1 ? curveOf(entries[0]) : null;
  return blendEntries(entries, (e) => e.focal, focal, curveOf);
}

/**
 * Vignetting entries at the nearest calibrated focal length (an unknown one needs all entries to share a single
 * focal length), one per aperture (distance nearest 1000 m), blended in log2 f-number; none without a known f-number.
 */
function vignettingByAperture(entries: LfVignetting[], focal: number, fNumber: number, scale: number): Curve | null {
  if (entries.length === 0 || !isKnown(fNumber)) return null;
  let nearest = entries[0].focal;
  if (isKnown(focal)) {
    for (const e of entries) if (Math.abs(e.focal - focal) < Math.abs(nearest - focal)) nearest = e.focal;
  } else if (entries.some((e) => e.focal !== nearest)) {
    return null;
  }
  const perAperture = new Map<number, LfVignetting>();
  for (const e of entries) {
    if (e.focal !== nearest) continue;
    const kept = perAperture.get(e.aperture);
    if (!kept || Math.abs(e.distance - 1000) < Math.abs(kept.distance - 1000)) perAperture.set(e.aperture, e);
  }
  return blendEntries([...perAperture.values()], (e) => Math.log2(e.aperture), Math.log2(fNumber), (e) => (rho) => evalVignettingGain(e, rho * scale));
}

/** The matched lens's calibration at this focal length and f-number, as curves in the renderer's radius unit. */
export function lensfunCurves(profile: LensProfile, focal: number, fNumber: number): LensCurves {
  const units = huginUnits(profile);
  return {
    distortion: byFocal(profile.distortion, focal, (e) => (rho) => distortionRatio(e, rho * units.geometry)),
    caRed: byFocal(profile.tca, focal, (e) => (rho) => tcaRatio(e, rho * units.geometry).red),
    caBlue: byFocal(profile.tca, focal, (e) => (rho) => tcaRatio(e, rho * units.geometry).blue),
    vignetting: vignettingByAperture(profile.vignetting, focal, fNumber, units.vignetting),
  };
}

// ── Manual sliders ──────────────────────────────────────────────────

/** Curves for the manual sliders; a slider at its default gives a null curve. `halfDiagPx` converts the CA pixels at the corner. */
export function manualCurves(manual: LensManual, halfDiagPx: number): LensCurves {
  const p = Math.log(0.5) / Math.log(manual.midpoint);
  const caFactor = (px: number) => (px === 0 ? null : () => 1 + px / halfDiagPx);
  return {
    distortion: manual.distortion === 0 ? null : (rho) => 1 - (manual.distortion / 100) * rho * rho,
    caRed: caFactor(manual.caRed),
    caBlue: caFactor(manual.caBlue),
    vignetting: manual.vignetting === 0 ? null : (rho) => 2 ** (manual.vignetting * rho ** p),
  };
}
