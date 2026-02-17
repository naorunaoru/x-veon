/**
 * Compute correlated color temperature (CCT) and tint from
 * white balance coefficients and camera-to-XYZ matrix.
 *
 * The illuminant's camera-space color is inversely proportional to the
 * WB multipliers. We transform to XYZ, derive CIE 1931 chromaticity,
 * then use McCamy's approximation for CCT and Planckian locus distance
 * for tint (green/magenta deviation).
 */

/** Kang et al. (2002) Planckian locus x(T) approximation */
function planckianX(T: number): number {
  const T2 = T * T;
  const T3 = T2 * T;
  if (T <= 4000) {
    return -0.2661239e9 / T3 - 0.2343589e6 / T2 + 0.8776956e3 / T + 0.179910;
  }
  return -3.0258469e9 / T3 + 2.1070379e6 / T2 + 0.2226347e3 / T + 0.240390;
}

/** Kang et al. (2002) Planckian locus y(x) approximation */
function planckianY(x: number, T: number): number {
  const x2 = x * x;
  const x3 = x2 * x;
  if (T <= 2222) {
    return -1.1063814 * x3 - 1.3481102 * x2 + 2.18555832 * x - 0.20219683;
  }
  if (T <= 4000) {
    return -0.9549476 * x3 - 1.37418593 * x2 + 2.09137015 * x - 0.16748867;
  }
  return 3.081758 * x3 - 5.8733867 * x2 + 3.75112997 * x - 0.37001483;
}

/** Convert CIE 1931 (x,y) to CIE 1960 UCS (u,v) */
function xyToUv(x: number, y: number): [number, number] {
  const d = -2 * x + 12 * y + 3;
  return [4 * x / d, 6 * y / d];
}

/** Core CCT/tint estimation without rounding (for numerical use in bisection). */
function estimateRaw(
  wb: Float32Array,
  camToXyz: Float32Array,
): { cct: number; tint: number } {
  const illR = 1.0 / wb[0];
  const illG = 1.0 / wb[1];
  const illB = 1.0 / wb[2];

  const X = camToXyz[0] * illR + camToXyz[1] * illG + camToXyz[2] * illB;
  const Y = camToXyz[4] * illR + camToXyz[5] * illG + camToXyz[6] * illB;
  const Z = camToXyz[8] * illR + camToXyz[9] * illG + camToXyz[10] * illB;

  const sum = X + Y + Z;
  if (sum <= 0) return { cct: 5500, tint: 0 };

  const x = X / sum;
  const y = Y / sum;

  const n = (x - 0.3320) / (0.1858 - y);
  let cct = 449 * n * n * n + 3525 * n * n + 6823.3 * n + 5520.33;
  cct = Math.max(1500, Math.min(25000, cct));

  const [u, v] = xyToUv(x, y);
  const px = planckianX(cct);
  const py = planckianY(px, cct);
  const [pu, pv] = xyToUv(px, py);

  const du = u - pu;
  const dv = v - pv;
  const duv = Math.sqrt(du * du + dv * dv);

  const px2 = planckianX(cct + 1);
  const py2 = planckianY(px2, cct + 1);
  const [pu2, pv2] = xyToUv(px2, py2);
  const tx = pu2 - pu;
  const ty = pv2 - pv;
  const cross = tx * dv - ty * du;
  const signedDuv = cross >= 0 ? duv : -duv;

  return { cct, tint: signedDuv * 3200 };
}

/**
 * Estimate CCT and tint from camera white balance coefficients and color matrix.
 *
 * @param wb  WB coefficients [R, G, B] (G is typically 1.0 but may differ with tint adjustment)
 * @param camToXyz  Camera→XYZ matrix: 3×4 flattened row-major (12 elements, stride 4)
 * @returns { temp, tint } where temp is in Kelvin (rounded to 50K) and
 *          tint is a signed value (positive = magenta, negative = green),
 *          scaled to roughly ±150 range like common photo editors.
 */
export function estimateColorTemperature(
  wb: Float32Array,
  camToXyz: Float32Array,
): { temp: number; tint: number } {
  const raw = estimateRaw(wb, camToXyz);
  return {
    temp: Math.round(raw.cct / 50) * 50,
    tint: Math.round(raw.tint),
  };
}

/**
 * Find the wb_temp parameter value that produces a target CCT.
 * Uses brute-force sampling because McCamy's CCT approximation is not
 * monotonic over the full wb_temp range (it breaks down at extreme
 * chromaticities), making bisection unreliable.
 */
export function findWbTempForCct(
  targetCct: number,
  baseWb: Float32Array,
  camToXyz: Float32Array,
): number {
  const N = 400;
  let bestT = 0, bestScore = Infinity;
  for (let i = 0; i <= N; i++) {
    const t = -4 + 8 * i / N;
    const wb = new Float32Array([baseWb[0] * 2 ** t, 1, baseWb[2] * 2 ** (-t)]);
    const { temp } = estimateColorTemperature(wb, camToXyz);
    const diff = Math.abs(temp - targetCct);
    // Penalize extreme t values to avoid non-monotonic CCT artifacts —
    // McCamy's approximation can produce matching CCTs at wildly different
    // chromaticities, so prefer the solution closest to no-adjustment (t=0).
    const score = diff + Math.abs(t) * 100;
    if (score < bestScore) { bestScore = score; bestT = t; }
  }
  return bestT;
}

/**
 * Find the wb_tint parameter value that produces a target tint.
 * Same brute-force approach as findWbTempForCct — the tint-vs-wb_tint
 * relationship can also be non-monotonic at extreme values.
 */
export function findWbTintForTint(
  targetTint: number,
  baseWb: Float32Array,
  camToXyz: Float32Array,
): number {
  const N = 400;
  let bestT = 0, bestDiff = Infinity;
  for (let i = 0; i <= N; i++) {
    const t = -4 + 8 * i / N;
    const wb = new Float32Array([baseWb[0], 2 ** (-t), baseWb[2]]);
    const { tint } = estimateColorTemperature(wb, camToXyz);
    const diff = Math.abs(tint - targetTint);
    if (diff < bestDiff) { bestDiff = diff; bestT = t; }
  }
  return bestT;
}
