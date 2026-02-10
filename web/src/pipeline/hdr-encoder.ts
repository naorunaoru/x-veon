import { SRGB_TO_BT2020 } from './constants';
import { buildColorMatrix, applyExifRotation, mul3x3 } from './postprocessor';

function buildHdrColorMatrix(xyzToCam3x3: Float32Array): Float32Array {
  const camToSrgb = buildColorMatrix(xyzToCam3x3);
  return mul3x3(new Float32Array(SRGB_TO_BT2020), camToSrgb);
}

function applyHdrColorCorrection(
  hwc: Float32Array, numPixels: number, combinedMatrix: Float32Array,
): void {
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

function applyHlgOetf(hwc: Float32Array, numPixels: number): void {
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

function toHdrImageData(hwc: Float32Array, width: number, height: number): ImageData {
  const n = width * height;

  let imageData: ImageData;
  try {
    imageData = new ImageData(width, height, {
      colorSpace: 'rec2100-hlg' as PredefinedColorSpace,
      storageFormat: 'float32',
    } as ImageDataSettings);
  } catch {
    imageData = new ImageData(width, height, {
      colorSpace: 'rec2100-hlg' as PredefinedColorSpace,
    } as ImageDataSettings);
  }

  const rgba = imageData.data;
  const isFloat = rgba instanceof Float32Array;

  for (let i = 0; i < n; i++) {
    const si = i * 3;
    const di = i * 4;
    if (isFloat) {
      (rgba as unknown as Float32Array)[di]     = hwc[si];
      (rgba as unknown as Float32Array)[di + 1] = hwc[si + 1];
      (rgba as unknown as Float32Array)[di + 2] = hwc[si + 2];
      (rgba as unknown as Float32Array)[di + 3] = 1.0;
    } else {
      rgba[di]     = (Math.min(1, Math.max(0, hwc[si]))     * 255 + 0.5) | 0;
      rgba[di + 1] = (Math.min(1, Math.max(0, hwc[si + 1])) * 255 + 0.5) | 0;
      rgba[di + 2] = (Math.min(1, Math.max(0, hwc[si + 2])) * 255 + 0.5) | 0;
      rgba[di + 3] = 255;
    }
  }

  console.log(`HDR ImageData: ${isFloat ? 'float32' : 'uint8'}, colorSpace=${imageData.colorSpace}`);
  return imageData;
}

export function processHdr(
  hwc: Float32Array, numPixels: number, xyzToCam3x3: Float32Array | null,
  width: number, height: number, orientation: string,
): ImageData {
  if (xyzToCam3x3) {
    const hdrMatrix = buildHdrColorMatrix(xyzToCam3x3);
    applyHdrColorCorrection(hwc, numPixels, hdrMatrix);
  }

  const rotated = applyExifRotation(hwc, width, height, orientation);
  applyHlgOetf(rotated.data, rotated.width * rotated.height);
  return toHdrImageData(rotated.data, rotated.width, rotated.height);
}
