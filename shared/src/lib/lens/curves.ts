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
