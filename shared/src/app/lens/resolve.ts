import { lensfunCurves, manualCurves, NO_CURVES, type LensCurves } from '@/lib/lens/curves';
import type { LensEdit } from '@/lib/lens/edit';
import { fujiCurves, type FujiKnots } from '@/lib/lens/fuji-tables';
import type { LensProfile } from '@/lib/lensfun-types';

export type LensSourceKind = 'camera' | 'lensfun' | 'manual' | 'none';
type Correction = 'distortion' | 'ca' | 'vignetting';
type Sources = Record<Correction, LensSourceKind>;

export interface ResolvedLens {
  curves: LensCurves;
  /** Where each correction comes from. */
  sources: Sources;
  /** The manual sliders are the correction in force, so the panel shows them live. */
  manualLive: boolean;
}

const CORRECTIONS: readonly Correction[] = ['distortion', 'ca', 'vignetting'];
/** CA is one correction with a curve per channel; the sources of its curves are never mixed. */
const CURVES_OF: Record<Correction, readonly (keyof LensCurves)[]> = {
  distortion: ['distortion'],
  ca: ['caRed', 'caBlue'],
  vignetting: ['vignetting'],
};

const has = (curves: LensCurves, correction: Correction) => CURVES_OF[correction].some((name) => curves[name] !== null);

/** Per correction, the first candidate that has it. */
function firstWith(candidates: readonly (readonly [LensSourceKind, LensCurves])[]): Pick<ResolvedLens, 'curves' | 'sources'> {
  const curves: LensCurves = { ...NO_CURVES };
  const sources: Sources = { distortion: 'none', ca: 'none', vignetting: 'none' };
  for (const correction of CORRECTIONS) {
    const found = candidates.find(([, c]) => has(c, correction));
    if (!found) continue;
    sources[correction] = found[0];
    for (const name of CURVES_OF[correction]) curves[name] = found[1][name];
  }
  return { curves, sources };
}

/** A profile stored before the calibration fields existed cannot be evaluated: it counts as no profile. */
const usable = (profile: LensProfile | null | undefined): profile is LensProfile => typeof profile?.calibCropfactor === 'number';

/**
 * Decides which source provides each correction. Auto takes the camera's tables where they are non-trivial, else the
 * automatic lensfun match; a hand-picked lens uses lensfun only (`picked` is undefined until its profile is loaded);
 * manual uses the sliders only. The sliders are also in force when Auto finds nothing at all.
 */
export function resolveLens(args: {
  edit: LensEdit;
  camera: FujiKnots | null;
  lensfun: LensProfile | null;
  picked: LensProfile | null | undefined;
  focal: number;
  fNumber: number;
  halfDiagPx: number;
}): ResolvedLens {
  const { edit, camera, lensfun, picked, focal, fNumber, halfDiagPx } = args;
  const fromLensfun = (profile: LensProfile | null | undefined) =>
    ['lensfun', usable(profile) ? lensfunCurves(profile, focal, fNumber) : NO_CURVES] as const;
  const { source } = edit;

  let resolved: Pick<ResolvedLens, 'curves' | 'sources'>;
  if (source === 'manual') resolved = firstWith([]);
  else if (source === 'auto') resolved = firstWith([['camera', camera ? fujiCurves(camera) : NO_CURVES], fromLensfun(lensfun)]);
  else resolved = firstWith([fromLensfun(picked)]);

  const manualLive = source === 'manual' || (source === 'auto' && CORRECTIONS.every((c) => resolved.sources[c] === 'none'));
  return manualLive
    ? { ...firstWith([['manual', manualCurves(edit.manual, halfDiagPx)]]), manualLive }
    : { ...resolved, manualLive };
}

/** What the corrections do at the corner of the frame (ρ = 1). */
export interface LensReadouts {
  /** % of the radius the corner moves (positive pulls it in) */
  distortion: number;
  /** px the red and blue channels sit off green at the corner */
  caRed: number;
  caBlue: number;
  /** EV of the vignetting gain at the corner */
  vignetting: number;
}

export function lensReadouts(curves: LensCurves, halfDiagPx: number): LensReadouts {
  const mG = curves.distortion?.(1) ?? 1;
  const offGreen = (ca: LensCurves['caRed']) => (mG * (ca?.(mG) ?? 1) - mG) * halfDiagPx;
  return {
    distortion: (1 - mG) * 100,
    caRed: offGreen(curves.caRed),
    caBlue: offGreen(curves.caBlue),
    vignetting: Math.log2(curves.vignetting?.(mG) ?? 1),
  };
}

/** The panel's status line and, when the sources differ per correction, a detail line. */
export function lensStatus(args: {
  resolved: ResolvedLens | null;
  edit: LensEdit;
  exifLens: string;
  lensfunModel: string | null;
  matching: boolean;
}): { line: string; detail: string | null } {
  const { resolved, edit, exifLens, lensfunModel, matching } = args;
  const named = (prefix: string, joiner: string, name: string) => (name ? `${prefix}${joiner}${name}` : prefix);
  const line = (text: string) => ({ line: text, detail: null });

  if (!resolved) return line('Waiting for decode');
  if (matching && edit.source === 'auto') return line('Matching lens');
  if (edit.source === 'manual') return line('Manual');
  if (typeof edit.source === 'object') return line(`Lensfun · ${lensfunModel ?? edit.source.lensfun.model}`);

  const used = new Set(CORRECTIONS.map((c) => resolved.sources[c]).filter((s) => s !== 'none'));
  if (resolved.manualLive) return line(named('No profile', ' for ', exifLens));
  if (used.size === 2) {
    const { distortion, ca, vignetting } = resolved.sources;
    return { line: 'Camera + Lensfun', detail: `Distortion: ${distortion} · CA: ${ca} · Vignetting: ${vignetting}` };
  }
  return line(used.has('camera') ? named('Camera', ' · ', exifLens) : named('Lensfun', ' · ', lensfunModel ?? ''));
}
