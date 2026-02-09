// X-Trans 6x6 CFA pattern: R=0, G=1, B=2
export const XTRANS_PATTERN = [
  [0, 2, 1, 2, 0, 1],
  [1, 1, 0, 1, 1, 2],
  [1, 1, 2, 1, 1, 0],
  [2, 0, 1, 0, 2, 1],
  [1, 1, 2, 1, 1, 0],
  [1, 1, 0, 1, 1, 2],
];

// XYZ to sRGB (D65 whitepoint)
export const XYZ_TO_SRGB = [
   3.2404542, -1.5371385, -0.4985314,
  -0.9692660,  1.8760108,  0.0415560,
   0.0556434, -0.2040259,  1.0572252,
];

export const PATCH_SIZE = 288;
export const OVERLAP = 48;
