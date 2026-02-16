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
