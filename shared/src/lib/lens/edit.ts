/** Where a photo's lens profile comes from: matched from EXIF, hand-set, or a lensfun lens the user picked. */
export type LensSource = 'auto' | 'manual' | { lensfun: { file: string; model: string } };

export interface LensManual {
  distortion: number;
  caRed: number;
  caBlue: number;
  vignetting: number;
  midpoint: number;
}

export interface LensEdit {
  source: LensSource;
  distortion: boolean;
  ca: boolean;
  vignetting: boolean;
  manual: LensManual;
}

export const DEFAULT_LENS_EDIT: Readonly<LensEdit> = Object.freeze({
  source: 'auto',
  distortion: true,
  ca: true,
  vignetting: true,
  manual: Object.freeze({ distortion: 0, caRed: 0, caBlue: 0, vignetting: 0, midpoint: 0.5 }),
});

/** A copy of the photo's lens block with defaults filled in; an absent block means the defaults. */
export function effectiveLensEdit(edit: { lens?: LensEdit }): LensEdit {
  return { ...DEFAULT_LENS_EDIT, ...edit.lens, manual: { ...DEFAULT_LENS_EDIT.manual, ...edit.lens?.manual } };
}

export function lensSourceKey(source: LensSource): string {
  return typeof source === 'string' ? source : `lensfun:${source.lensfun.file}#${source.lensfun.model}`;
}

/** Compares effective values, so an absent block equals one spelled out with the defaults. */
export function lensEditEquals(a: { lens?: LensEdit }, b: { lens?: LensEdit }): boolean {
  const x = effectiveLensEdit(a);
  const y = effectiveLensEdit(b);
  return lensSourceKey(x.source) === lensSourceKey(y.source)
    && x.distortion === y.distortion && x.ca === y.ca && x.vignetting === y.vignetting
    && x.manual.distortion === y.manual.distortion && x.manual.caRed === y.manual.caRed
    && x.manual.caBlue === y.manual.caBlue && x.manual.vignetting === y.manual.vignetting
    && x.manual.midpoint === y.manual.midpoint;
}

export function isLensEditModified(edit: { lens?: LensEdit }): boolean {
  return !lensEditEquals(edit, {});
}
