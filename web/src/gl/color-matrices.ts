// Precomputed color space conversion matrices for WebGL preview.
// All matrices stored in row-major order as flat Float32Array (for gl.uniformMatrix3fv).

// sRGB → P3-D65 = XYZ_TO_P3D65 * inv(XYZ_TO_SRGB)
// Since hwc data is already color-corrected to linear sRGB, this is needed
// to get into the P3-D65 working space for OpenDRT.
// Near-zero off-diagonals zeroed out (they're ~1e-8 floating-point noise —
// sRGB and P3 share D65 white point so rows sum to 1.0).
export const SRGB_TO_P3D65 = new Float32Array([
   0.8225929,  0.1775339,  0.0,
   0.0331995,  0.9667835,  0.0,
   0.0170854,  0.0723958,  0.9103015,
]);

// P3-D65 → Rec.709 (from opendrt.rs P3D65_TO_REC709)
export const P3D65_TO_REC709 = new Float32Array([
   1.224940181, -0.2249402404,  0.0,
  -0.04205697775, 1.042057037, 0.0,
  -0.01963755488,-0.07863604277, 1.098273635,
]);

// P3-D65 → Rec.2020 (from opendrt.rs P3D65_TO_REC2020, for future HDR)
export const P3D65_TO_REC2020 = new Float32Array([
  0.7538330344, 0.1985973691, 0.04756959659,
  0.04574384897, 0.9417772198, 0.01247893122,
 -0.001210340355, 0.0176017173, 0.9836086231,
]);

// Identity 3×3 — used when HDR display mode keeps output in P3 (no gamut conversion).
export const IDENTITY_3X3 = new Float32Array([
  1, 0, 0,
  0, 1, 0,
  0, 0, 1,
]);

// Creative White adaptation matrices (CAT02, from OpenDRT CTL).
// These convert P3-D65 to display gamut with D50 white point adaptation.

// P3-D65 → Rec.709 with D50 adaptation
export const P3D65_TO_REC709_D50 = new Float32Array([
   1.103807322,   -0.1103425121,  0.006531676079,
  -0.04079386701,  0.8704694227, -0.000180522628,
  -0.01854055914, -0.07857582481, 0.7105498861,
]);

// P3-D65 → P3-D65 with D50 adaptation (for HDR P3 output)
export const P3D65_TO_P3D65_D50 = new Float32Array([
  0.9287127388,  0.06578032793, 0.005506708345,
 -0.002887159176, 0.8640709228,  4.3593718e-05,
 -0.001009551548,-0.01073503317, 0.6672692039,
]);

/** Compute CWP adaptation matrix: cwp_adapt = P3→display_cwp * inv(P3→display_D65).
 *  When applied to D65 display-referred RGB, gives the D50-adapted result. */
export function computeCwpAdaptMatrix(isHdr: boolean): Float32Array {
  const d65 = isHdr ? IDENTITY_3X3 : P3D65_TO_REC709;
  const d50 = isHdr ? P3D65_TO_P3D65_D50 : P3D65_TO_REC709_D50;
  const inv = invert3x3(d65);
  return multiply3x3(d50, inv);
}

// ── 3×3 matrix math (row-major) ──────────────────────────────────────────

function invert3x3(m: Float32Array): Float32Array {
  const [a, b, c, d, e, f, g, h, i] = m;
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  const invDet = 1.0 / det;
  return new Float32Array([
    (e * i - f * h) * invDet, (c * h - b * i) * invDet, (b * f - c * e) * invDet,
    (f * g - d * i) * invDet, (a * i - c * g) * invDet, (c * d - a * f) * invDet,
    (d * h - e * g) * invDet, (b * g - a * h) * invDet, (a * e - b * d) * invDet,
  ]);
}

function multiply3x3(a: Float32Array, b: Float32Array): Float32Array {
  const r = new Float32Array(9);
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 3; col++) {
      r[row * 3 + col] =
        a[row * 3] * b[col] +
        a[row * 3 + 1] * b[3 + col] +
        a[row * 3 + 2] * b[6 + col];
    }
  }
  return r;
}
