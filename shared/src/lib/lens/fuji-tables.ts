import { linearSpline, type LensCurves } from './curves';

/** The three correction tags of a Fuji RAF, as the decoder read them. */
export interface FujiTables { distortion: number[]; ca: number[]; vignetting: number[] }

/** The tables split into per-knot columns; `knots` are source radii in half-diagonal units. */
export interface FujiKnots {
  knots: number[];
  distortion: number[];
  caRed: number[];
  caBlue: number[];
  vignetting: number[];
  /** tables[0] × knot count: the camera's half-diagonal in pixels; 0 when the tag's first value is 0 */
  halfDiagPx: number;
}

/**
 * The two tag layouts, told apart by element counts. Distortion and vignetting hold [scale, n knots,
 * n values]; CA holds [scale, c knots, c red, c blue], plus a trailing scale in the new layout.
 * The old layout's CA knots skip the zero knot the others start with (c = n - 1).
 */
const LAYOUTS = [
  { distortionCount: 19, caCount: 29, vignettingCount: 19, n: 9, c: 9 },    // X-Trans IV, V
  { distortionCount: 23, caCount: 31, vignettingCount: 23, n: 11, c: 10 },  // X-Trans I, II, III
];

const KNOT_TOLERANCE = 1e-6;

function sameKnots(a: number[], b: number[]): boolean {
  return a.length === b.length && a.every((v, i) => Math.abs(v - b[i]) <= KNOT_TOLERANCE);
}

/** Null when the counts match neither layout, or the three tags do not share their knot positions. */
export function parseFujiTables(t: FujiTables): FujiKnots | null {
  const layout = LAYOUTS.find(
    (l) => t.distortion.length === l.distortionCount && t.ca.length === l.caCount && t.vignetting.length === l.vignettingCount,
  );
  if (!layout) return null;
  const { n, c } = layout;

  const knots = t.distortion.slice(1, 1 + n);
  if (!sameKnots(knots, t.vignetting.slice(1, 1 + n))) return null;
  if (!sameKnots(knots.slice(n - c), t.ca.slice(1, 1 + c))) return null;

  const zeroKnots = new Array<number>(n - c).fill(0); // no CA at the centre
  return {
    knots,
    distortion: t.distortion.slice(1 + n, 1 + 2 * n),
    caRed: [...zeroKnots, ...t.ca.slice(1 + c, 1 + 2 * c)],
    caBlue: [...zeroKnots, ...t.ca.slice(1 + 2 * c, 1 + 3 * c)],
    vignetting: t.vignetting.slice(1 + n, 1 + 2 * n),
    halfDiagPx: t.distortion[0] * n,
  };
}

/** Which corrections are non-trivial: distortion and CA not all zero, vignetting not all 100. */
export function fujiHas(k: FujiKnots): { distortion: boolean; ca: boolean; vignetting: boolean } {
  return {
    distortion: k.distortion.some((d) => d !== 0),
    ca: k.caRed.some((v) => v !== 0) || k.caBlue.some((v) => v !== 0),
    vignetting: k.vignetting.some((v) => v !== 100),
  };
}

/** A spline through the knots that is `1` (no correction) at the centre when the first knot is not there. */
function fromCentre(xs: number[], ys: number[]): (x: number) => number {
  return xs[0] > 0 ? linearSpline([0, ...xs], [1, ...ys]) : linearSpline(xs, ys);
}

/**
 * Curves with darktable's second algorithm (`_init_coeffs_md_v2`). The knots are source radii; the
 * distortion multiplier m = 1 + d/100 gives source = m × output, so the distortion curve is
 * re-indexed to the output radius r_s / m. Parts that `has` marks trivial come out null.
 */
export function fujiCurves(k: FujiKnots, has = fujiHas(k)): LensCurves {
  const m = k.distortion.map((d) => 1 + d / 100);
  return {
    distortion: has.distortion ? fromCentre(k.knots.map((r, i) => r / m[i]), m) : null,
    caRed: has.ca ? fromCentre(k.knots, k.caRed.map((v) => 1 + v)) : null,
    caBlue: has.ca ? fromCentre(k.knots, k.caBlue.map((v) => 1 + v)) : null,
    vignetting: has.vignetting ? fromCentre(k.knots, k.vignetting.map((v) => 100 / v)) : null,
  };
}

/**
 * The camera's crop-rectangle centre in texture pixels. `frame` is [full height, full width, crop top,
 * crop left, cropped height, cropped width]. Null when it is absent, empty, or describes a readout of
 * another size than the decode (then the texture's own centre is the better guess).
 */
export function fujiFrameCentre(
  frame: number[],
  rawWidth: number,
  rawHeight: number,
  visible: { left: number; top: number; width: number; height: number },
): [number, number] | null {
  if (frame.length !== 6) return null;
  const [fullHeight, fullWidth, top, left, cropHeight, cropWidth] = frame;
  if (fullWidth !== rawWidth || fullHeight !== rawHeight || !(cropWidth > 0) || !(cropHeight > 0)) return null;
  return [left + cropWidth / 2 - visible.left, top + cropHeight / 2 - visible.top];
}
