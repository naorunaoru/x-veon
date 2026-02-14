// X-Trans 6x6 CFA pattern: R=0, G=1, B=2
export const XTRANS_PATTERN: readonly (readonly number[])[] = [
  [0, 2, 1, 2, 0, 1],
  [1, 1, 0, 1, 1, 2],
  [1, 1, 2, 1, 1, 0],
  [2, 0, 1, 0, 2, 1],
  [1, 1, 2, 1, 1, 0],
  [1, 1, 0, 1, 1, 2],
];

// Canonical Bayer 2x2 pattern (RGGB): R=0, G=1, B=2
export const BAYER_PATTERN: readonly (readonly number[])[] = [
  [0, 1],
  [1, 2],
];

export const RAW_EXTENSIONS = [
  '.raf', '.cr2', '.cr3', '.nef', '.nrw', '.arw', '.dng',
  '.rw2', '.orf', '.pef', '.srw', '.erf', '.kdc', '.dcr', '.mef',
];

// XYZ to sRGB (D65 whitepoint)
export const XYZ_TO_SRGB = [
   3.2404542, -1.5371385, -0.4985314,
  -0.9692660,  1.8760108,  0.0415560,
   0.0556434, -0.2040259,  1.0572252,
] as const;

// sRGB to BT.2020 (precomputed: XYZ_TO_BT2020 @ inv(XYZ_TO_SRGB))
export const SRGB_TO_BT2020 = [
  0.6274039,  0.3292830,  0.0433131,
  0.0690973,  0.9195404,  0.0113623,
  0.0163914,  0.0880133,  0.8955953,
] as const;

export const PATCH_SIZE = 288;
export const OVERLAP = 48;
