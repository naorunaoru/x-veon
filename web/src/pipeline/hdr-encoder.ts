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

/** Smoothstep: 0 at edge0, 1 at edge1, smooth Hermite interpolation between. */
function smoothstep(x: number, edge0: number, edge1: number): number {
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

/**
 * Fused HDR pipeline: highlight desaturation (camera space) + color correction + HLG + rotation.
 * Desaturates towards camera-space luminance before the color matrix, preventing the matrix's
 * negative off-diagonals from amplifying channel imbalances into magenta fringes.
 * Reads hwc without mutation so the buffer can be reused as export data.
 */
export function processHdr(
  hwc: Float32Array, numPixels: number, xyzToCam3x3: Float32Array | null,
  wb: Float32Array | null,
  width: number, height: number, orientation: string,
): ImageData {
  const combinedMatrix = xyzToCam3x3 ? buildHdrColorMatrix(xyzToCam3x3) : null;
  const wbR = wb ? wb[0] : 1, wbG = wb ? wb[1] : 1, wbB = wb ? wb[2] : 1;

  const swap = orientation === 'Rotate90' || orientation === 'Rotate270';
  const outW = swap ? height : width;
  const outH = swap ? width : height;

  // Pass 1: highlight desaturation (camera space) + color correction + HLG, track peak
  const hlgBuf = new Float32Array(numPixels * 3);
  let peak = 0;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const si = (y * width + x) * 3;
      let r = hwc[si], g = hwc[si + 1], b = hwc[si + 2];

      // Highlight desaturation toward mean (hue-preserving) in WB'd camera space.
      // Dual gate: clip proximity × neutralness. Mean target moves straight
      // toward neutral in chromaticity space, so partial gating is safe.
      const sR = r / wbR, sG = g / wbG, sB = b / wbB;
      const clipProx = Math.max(sR, sG, sB);
      const minSensor = Math.min(sR, sG, sB);
      const neutralness = clipProx > 1e-6 ? minSensor / clipProx : 1;
      const t = smoothstep(clipProx, 0.8, 1.0) * smoothstep(neutralness, 0.25, 0.45);
      if (t > 0) {
        const L = (r + g + b) / 3;
        r = r + t * (L - r);
        g = g + t * (L - g);
        b = b + t * (L - b);
      }

      if (combinedMatrix) {
        const or = r, og = g, ob = b;
        r = combinedMatrix[0] * or + combinedMatrix[1] * og + combinedMatrix[2] * ob;
        g = combinedMatrix[3] * or + combinedMatrix[4] * og + combinedMatrix[5] * ob;
        b = combinedMatrix[6] * or + combinedMatrix[7] * og + combinedMatrix[8] * ob;
      }

      r = Math.max(0, r);
      g = Math.max(0, g);
      b = Math.max(0, b);

      const rH = linearToHlg(r);
      const gH = linearToHlg(g);
      const bH = linearToHlg(b);

      hlgBuf[si]     = rH;
      hlgBuf[si + 1] = gH;
      hlgBuf[si + 2] = bH;

      const m = Math.max(rH, gH, bH);
      if (m > peak) peak = m;
    }
  }

  const scale = peak > 1.0 ? 1.0 / peak : 1.0;

  // Pass 2: normalize + rotate + write to ImageData
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
      const rH = hlgBuf[si]     * scale;
      const gH = hlgBuf[si + 1] * scale;
      const bH = hlgBuf[si + 2] * scale;

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
        rgba[di]     = (rH * 255 + 0.5) | 0;
        rgba[di + 1] = (gH * 255 + 0.5) | 0;
        rgba[di + 2] = (bH * 255 + 0.5) | 0;
        rgba[di + 3] = 255;
      }
    }
  }

  if (peak > 1.0) console.log(`HLG peak ${peak.toFixed(3)} → normalized by ${scale.toFixed(4)}`);
  console.log(`HDR ImageData: ${isFloat ? 'float32' : 'uint8'}, colorSpace=${imageData.colorSpace}`);
  return imageData;
}
