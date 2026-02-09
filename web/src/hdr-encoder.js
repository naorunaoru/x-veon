import { SRGB_TO_BT2020 } from './constants.js';
import { buildColorMatrix, applyExifRotation, mul3x3 } from './postprocessor.js';

/**
 * Build Camera→BT.2020 color correction matrix.
 * Chains the dcraw Camera→sRGB matrix with sRGB→BT.2020.
 */
function buildHdrColorMatrix(xyzToCam3x3) {
  const camToSrgb = buildColorMatrix(xyzToCam3x3);
  return mul3x3(new Float32Array(SRGB_TO_BT2020), camToSrgb);
}

/**
 * Apply Camera→BT.2020 color correction with highlight blending (in-place).
 * Blends toward SRGB_TO_BT2020 (no camera correction, still in BT.2020)
 * for bright pixels to avoid color cast from clipped sensors.
 * Does NOT clip to [0,1] — values >1 are HDR content.
 */
function applyHdrColorCorrection(hwc, numPixels, combinedMatrix) {
  const simple = new Float32Array(SRGB_TO_BT2020);
  const blendLo = 0.8, blendHi = 1.5;
  const blendRange = blendHi - blendLo;

  for (let i = 0; i < numPixels; i++) {
    const idx = i * 3;
    const r = hwc[idx], g = hwc[idx + 1], b = hwc[idx + 2];

    const fr = combinedMatrix[0] * r + combinedMatrix[1] * g + combinedMatrix[2] * b;
    const fg = combinedMatrix[3] * r + combinedMatrix[4] * g + combinedMatrix[5] * b;
    const fb = combinedMatrix[6] * r + combinedMatrix[7] * g + combinedMatrix[8] * b;

    const sr = simple[0] * r + simple[1] * g + simple[2] * b;
    const sg = simple[3] * r + simple[4] * g + simple[5] * b;
    const sb = simple[6] * r + simple[7] * g + simple[8] * b;

    const maxCh = Math.max(r, g, b);
    const alpha = Math.min(1, Math.max(0, (maxCh - blendLo) / blendRange));

    hwc[idx]     = Math.max(0, fr + alpha * (sr - fr));
    hwc[idx + 1] = Math.max(0, fg + alpha * (sg - fg));
    hwc[idx + 2] = Math.max(0, fb + alpha * (sb - fb));
  }
}

/**
 * Apply BT.2100 HLG OETF in-place.
 * Input: linear scene-referred light (can exceed 1.0 for HDR).
 * Output: non-linear HLG signal.
 */
function applyHlgOetf(hwc, numPixels) {
  const a = 0.17883277;
  const b = 0.28466892;
  const c = 0.55991073;

  for (let i = 0; i < numPixels * 3; i++) {
    const E = Math.max(0, hwc[i]);
    hwc[i] = E <= 1 / 12
      ? Math.sqrt(3 * E)
      : a * Math.log(Math.max(12 * E - b, 1e-10)) + c;
  }
}

/**
 * Convert HLG float data to an HDR ImageData (rec2100-hlg).
 * Tries float32 storage first; falls back to uint8 if unsupported.
 */
function toHdrImageData(hwc, width, height) {
  const n = width * height;

  let imageData;
  try {
    imageData = new ImageData(width, height, {
      colorSpace: 'rec2100-hlg',
      storageFormat: 'float32',
    });
  } catch {
    imageData = new ImageData(width, height, {
      colorSpace: 'rec2100-hlg',
    });
  }

  const rgba = imageData.data;
  const isFloat = rgba instanceof Float32Array;

  for (let i = 0; i < n; i++) {
    const si = i * 3;
    const di = i * 4;
    if (isFloat) {
      rgba[di]     = hwc[si];
      rgba[di + 1] = hwc[si + 1];
      rgba[di + 2] = hwc[si + 2];
      rgba[di + 3] = 1.0;
    } else {
      // Uint8ClampedArray: scale HLG [0,1] → [0,255]
      rgba[di]     = (Math.min(1, Math.max(0, hwc[si]))     * 255 + 0.5) | 0;
      rgba[di + 1] = (Math.min(1, Math.max(0, hwc[si + 1])) * 255 + 0.5) | 0;
      rgba[di + 2] = (Math.min(1, Math.max(0, hwc[si + 2])) * 255 + 0.5) | 0;
      rgba[di + 3] = 255;
    }
  }

  console.log(`HDR ImageData: ${isFloat ? 'float32' : 'uint8'}, colorSpace=${imageData.colorSpace}`);
  return imageData;
}

/**
 * Process linear camera RGB into an HDR ImageData ready for canvas display.
 * Pipeline: BT.2020 color correction → EXIF rotation → HLG OETF → ImageData.
 *
 * @param {Float32Array} hwc - Linear camera RGB in HWC layout (mutated in-place)
 * @param {number} numPixels
 * @param {Float32Array|null} xyzToCam3x3
 * @param {number} width
 * @param {number} height
 * @param {string} orientation
 * @returns {ImageData} HDR ImageData with colorSpace 'rec2100-hlg'
 */
export function processHdr(hwc, numPixels, xyzToCam3x3, width, height, orientation) {
  if (xyzToCam3x3) {
    const hdrMatrix = buildHdrColorMatrix(xyzToCam3x3);
    applyHdrColorCorrection(hwc, numPixels, hdrMatrix);
  }

  const rotated = applyExifRotation(hwc, width, height, orientation);
  applyHlgOetf(rotated.data, rotated.width * rotated.height);
  return toHdrImageData(rotated.data, rotated.width, rotated.height);
}
