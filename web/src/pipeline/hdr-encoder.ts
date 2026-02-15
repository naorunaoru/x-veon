import { SRGB_TO_BT2020 } from './constants';

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
 * HDR display pipeline: sRGB→BT.2020 gamut + HLG OETF + rotation.
 * Color correction is applied upstream in useProcessFile.
 * Reads hwc without mutation so the buffer can be reused as export data.
 */
export function processHdr(
  hwc: Float32Array,
  width: number, height: number, orientation: string,
): ImageData {
  const srgbToBt2020 = new Float32Array(SRGB_TO_BT2020);

  const swap = orientation === 'Rotate90' || orientation === 'Rotate270';
  const outW = swap ? height : width;
  const outH = swap ? width : height;

  const hlgBuf = new Float32Array(width * height * 3);
  let peak = 0;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const si = (y * width + x) * 3;
      let r = hwc[si], g = hwc[si + 1], b = hwc[si + 2];

      // sRGB→BT.2020 gamut conversion
      {
        const pr = r, pg = g, pb = b;
        r = srgbToBt2020[0] * pr + srgbToBt2020[1] * pg + srgbToBt2020[2] * pb;
        g = srgbToBt2020[3] * pr + srgbToBt2020[4] * pg + srgbToBt2020[5] * pb;
        b = srgbToBt2020[6] * pr + srgbToBt2020[7] * pg + srgbToBt2020[8] * pb;
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
        di = (width * height - 1 - (y * width + x)) * 4;
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
  return imageData;
}
