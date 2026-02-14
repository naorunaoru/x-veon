import { SRGB_TO_BT2020 } from './constants';
import { buildColorMatrix, mul3x3 } from './postprocessor';

function buildHdrColorMatrix(xyzToCam3x3: Float32Array): Float32Array {
  const camToSrgb = buildColorMatrix(xyzToCam3x3);
  return mul3x3(new Float32Array(SRGB_TO_BT2020), camToSrgb);
}

// HLG OETF constants (BT.2100)
const HLG_A = 0.17883277;
const HLG_B = 0.28466892;
const HLG_C = 0.55991073;

function linearToHlg(v: number): number {
  v = Math.max(v, 0);
  if (v <= 1 / 12) return Math.sqrt(3 * v);
  return HLG_A * Math.log(Math.max(12 * v - HLG_B, 1e-10)) + HLG_C;
}

/**
 * Fused HDR pipeline: color correction + highlight blend + HLG + rotation.
 * Mirrors infer_hdr.py — scene-referred HLG, no OOTF (display-side).
 * Reads hwc without mutation so the buffer can be reused as export data.
 */
export function processHdr(
  hwc: Float32Array, numPixels: number, xyzToCam3x3: Float32Array | null,
  width: number, height: number, orientation: string,
): ImageData {
  const combinedMatrix = xyzToCam3x3 ? buildHdrColorMatrix(xyzToCam3x3) : null;
  const simple = combinedMatrix ? new Float32Array(SRGB_TO_BT2020) : null;
  const blendLo = 0.8, blendRange = 0.7;

  const swap = orientation === 'Rotate90' || orientation === 'Rotate270';
  const outW = swap ? height : width;
  const outH = swap ? width : height;

  let imageData: ImageData;
  try {
    imageData = new ImageData(outW, outH, {
      colorSpace: 'rec2100-hlg' as PredefinedColorSpace,
      storageFormat: 'float32',
    } as ImageDataSettings);
  } catch {
    imageData = new ImageData(outW, outH, {
      colorSpace: 'rec2100-hlg' as PredefinedColorSpace,
    } as ImageDataSettings);
  }

  const rgba = imageData.data;
  const isFloat = rgba instanceof Float32Array;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const si = (y * width + x) * 3;
      let r = hwc[si], g = hwc[si + 1], b = hwc[si + 2];

      // Color correction with highlight blend (mirrors infer_hdr.py)
      if (combinedMatrix && simple) {
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

      // HLG OETF (scene-referred, no OOTF)
      const rH = linearToHlg(r);
      const gH = linearToHlg(g);
      const bH = linearToHlg(b);

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
        (rgba as unknown as Float32Array)[di]     = rH;
        (rgba as unknown as Float32Array)[di + 1] = gH;
        (rgba as unknown as Float32Array)[di + 2] = bH;
        (rgba as unknown as Float32Array)[di + 3] = 1.0;
      } else {
        rgba[di]     = (Math.min(1, Math.max(0, rH)) * 255 + 0.5) | 0;
        rgba[di + 1] = (Math.min(1, Math.max(0, gH)) * 255 + 0.5) | 0;
        rgba[di + 2] = (Math.min(1, Math.max(0, bH)) * 255 + 0.5) | 0;
        rgba[di + 3] = 255;
      }
    }
  }

  console.log(`HDR ImageData: ${isFloat ? 'float32' : 'uint8'}, colorSpace=${imageData.colorSpace}`);
  return imageData;
}
