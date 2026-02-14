import { SRGB_TO_BT2020 } from './constants';
import { buildColorMatrix, mul3x3 } from './postprocessor';

function buildHdrColorMatrix(xyzToCam3x3: Float32Array): Float32Array {
  const camToSrgb = buildColorMatrix(xyzToCam3x3);
  return mul3x3(new Float32Array(SRGB_TO_BT2020), camToSrgb);
}

// ST 2084 (PQ) constants
const PQ_M1 = 2610 / 16384;
const PQ_M2 = (2523 / 4096) * 128;
const PQ_C1 = 3424 / 4096;
const PQ_C2 = (2413 / 4096) * 32;
const PQ_C3 = (2392 / 4096) * 32;

// HLG OOTF reference values (BT.2100)
const OOTF_GAMMA = 1.2;
const PEAK_NITS = 1000;

function linearToPq(nits: number): number {
  const y = Math.pow(nits / 10000, PQ_M1);
  return Math.pow((PQ_C1 + PQ_C2 * y) / (1 + PQ_C3 * y), PQ_M2);
}

/**
 * Fused HDR pipeline: color correction + rotation + OOTF + PQ + ImageData.
 * Reads hwc without mutation so the buffer can be reused as export data.
 */
export function processHdr(
  hwc: Float32Array, numPixels: number, xyzToCam3x3: Float32Array | null,
  width: number, height: number, orientation: string,
): ImageData {
  const combinedMatrix = xyzToCam3x3 ? buildHdrColorMatrix(xyzToCam3x3) : null;
  const simple = new Float32Array(SRGB_TO_BT2020);
  const blendLo = 0.8, blendRange = 0.7;
  const gammaMinusOne = OOTF_GAMMA - 1;

  const swap = orientation === 'Rotate90' || orientation === 'Rotate270';
  const outW = swap ? height : width;
  const outH = swap ? width : height;

  let imageData: ImageData;
  try {
    imageData = new ImageData(outW, outH, {
      colorSpace: 'rec2100-pq' as PredefinedColorSpace,
      storageFormat: 'float32',
    } as ImageDataSettings);
  } catch {
    imageData = new ImageData(outW, outH, {
      colorSpace: 'rec2100-pq' as PredefinedColorSpace,
    } as ImageDataSettings);
  }

  const rgba = imageData.data;
  const isFloat = rgba instanceof Float32Array;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const si = (y * width + x) * 3;
      let r = hwc[si], g = hwc[si + 1], b = hwc[si + 2];

      // HDR color correction with highlight blend
      if (combinedMatrix) {
        const fr = combinedMatrix[0] * r + combinedMatrix[1] * g + combinedMatrix[2] * b;
        const fg = combinedMatrix[3] * r + combinedMatrix[4] * g + combinedMatrix[5] * b;
        const fb = combinedMatrix[6] * r + combinedMatrix[7] * g + combinedMatrix[8] * b;
        const sr = simple[0] * r + simple[1] * g + simple[2] * b;
        const sg = simple[3] * r + simple[4] * g + simple[5] * b;
        const sb = simple[6] * r + simple[7] * g + simple[8] * b;
        const maxCh = Math.max(r, g, b);
        const alpha = Math.min(1, Math.max(0, (maxCh - blendLo) / blendRange));
        r = Math.max(0, fr + alpha * (sr - fr));
        g = Math.max(0, fg + alpha * (sg - fg));
        b = Math.max(0, fb + alpha * (sb - fb));
      }

      // OOTF + PQ
      r = Math.max(0, r); g = Math.max(0, g); b = Math.max(0, b);
      const Y = 0.2627 * r + 0.6780 * g + 0.0593 * b;
      const gain = Y > 0 ? Math.pow(Y, gammaMinusOne) * PEAK_NITS : 0;
      const rPq = linearToPq(r * gain);
      const gPq = linearToPq(g * gain);
      const bPq = linearToPq(b * gain);

      // Rotated destination index
      let di: number;
      if (orientation === 'Rotate180') {
        di = (numPixels - 1 - (y * width + x)) * 4;
      } else if (orientation === 'Rotate90') {
        di = (x * outW + (height - 1 - y)) * 4;
      } else if (orientation === 'Rotate270') {
        di = ((width - 1 - x) * outW + y) * 4;
      } else {
        di = (y * width + x) * 4;
      }

      if (isFloat) {
        (rgba as unknown as Float32Array)[di]     = rPq;
        (rgba as unknown as Float32Array)[di + 1] = gPq;
        (rgba as unknown as Float32Array)[di + 2] = bPq;
        (rgba as unknown as Float32Array)[di + 3] = 1.0;
      } else {
        rgba[di]     = (Math.min(1, Math.max(0, rPq)) * 255 + 0.5) | 0;
        rgba[di + 1] = (Math.min(1, Math.max(0, gPq)) * 255 + 0.5) | 0;
        rgba[di + 2] = (Math.min(1, Math.max(0, bPq)) * 255 + 0.5) | 0;
        rgba[di + 3] = 255;
      }
    }
  }

  console.log(`HDR ImageData: ${isFloat ? 'float32' : 'uint8'}, colorSpace=${imageData.colorSpace}`);
  return imageData;
}
