import { XYZ_TO_SRGB } from './constants';
import type { RotatedImage } from './types';

export function blendTiles(
  tileOutputs: Float32Array[], coords: Array<{ x: number; y: number }>,
  hPad: number, wPad: number, patchSize: number, overlap: number,
): Float32Array {
  const planeSize = hPad * wPad;
  const output = new Float32Array(3 * planeSize);
  const weights = new Float32Array(planeSize);

  const w1d = new Float32Array(patchSize);
  w1d.fill(1);
  for (let i = 0; i < overlap; i++) {
    w1d[i] = i / overlap;
    w1d[patchSize - 1 - i] = i / overlap;
  }

  const patchPixels = patchSize * patchSize;

  for (let t = 0; t < tileOutputs.length; t++) {
    const tile = tileOutputs[t];
    const { x: tx, y: ty } = coords[t];

    for (let py = 0; py < patchSize; py++) {
      const wy = w1d[py];
      const imgY = ty + py;
      for (let px = 0; px < patchSize; px++) {
        const w = wy * w1d[px];
        const imgIdx = imgY * wPad + (tx + px);
        const tileIdx = py * patchSize + px;

        for (let c = 0; c < 3; c++) {
          output[c * planeSize + imgIdx] += tile[c * patchPixels + tileIdx] * w;
        }
        weights[imgIdx] += w;
      }
    }
  }

  for (let i = 0; i < planeSize; i++) {
    if (weights[i] > 1e-8) {
      const invW = 1 / weights[i];
      for (let c = 0; c < 3; c++) {
        output[c * planeSize + i] *= invW;
      }
    }
  }

  return output;
}

export function cropToHWC(
  output: Float32Array, hPad: number, wPad: number,
  padTop: number, padLeft: number, hOrig: number, wOrig: number,
): Float32Array {
  const planeSize = hPad * wPad;
  const n = hOrig * wOrig;
  const hwc = new Float32Array(n * 3);

  for (let y = 0; y < hOrig; y++) {
    const srcRow = (y + padTop) * wPad + padLeft;
    const dstRow = y * wOrig;
    for (let x = 0; x < wOrig; x++) {
      const srcIdx = srcRow + x;
      const dstIdx = (dstRow + x) * 3;
      hwc[dstIdx]     = output[srcIdx];
      hwc[dstIdx + 1] = output[planeSize + srcIdx];
      hwc[dstIdx + 2] = output[2 * planeSize + srcIdx];
    }
  }

  return hwc;
}

export function invert3x3(m: Float32Array): Float32Array {
  const [a, b, c, d, e, f, g, h, i] = m;
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  const inv = 1 / det;
  return new Float32Array([
    (e * i - f * h) * inv, (c * h - b * i) * inv, (b * f - c * e) * inv,
    (f * g - d * i) * inv, (a * i - c * g) * inv, (c * d - a * f) * inv,
    (d * h - e * g) * inv, (b * g - a * h) * inv, (a * e - b * d) * inv,
  ]);
}

export function mul3x3(a: Float32Array, b: Float32Array): Float32Array {
  const r = new Float32Array(9);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      r[i * 3 + j] =
        a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
    }
  }
  return r;
}

export function buildColorMatrix(xyzToCam: Float32Array): Float32Array {
  const srgbToXyz = invert3x3(new Float32Array(XYZ_TO_SRGB));
  const srgbToCam = mul3x3(new Float32Array(xyzToCam), srgbToXyz);

  for (let i = 0; i < 3; i++) {
    const sum = srgbToCam[i * 3] + srgbToCam[i * 3 + 1] + srgbToCam[i * 3 + 2];
    srgbToCam[i * 3] /= sum;
    srgbToCam[i * 3 + 1] /= sum;
    srgbToCam[i * 3 + 2] /= sum;
  }

  return invert3x3(srgbToCam);
}

export function applyColorCorrection(
  hwc: Float32Array, numPixels: number, matrix: Float32Array,
): void {
  const blendLo = 0.8, blendHi = 1.5;
  const blendRange = blendHi - blendLo;

  for (let i = 0; i < numPixels; i++) {
    const idx = i * 3;
    const r = hwc[idx], g = hwc[idx + 1], b = hwc[idx + 2];

    const fr = matrix[0] * r + matrix[1] * g + matrix[2] * b;
    const fg = matrix[3] * r + matrix[4] * g + matrix[5] * b;
    const fb = matrix[6] * r + matrix[7] * g + matrix[8] * b;

    const maxCh = Math.max(r, g, b);
    const alpha = Math.min(1, Math.max(0, (maxCh - blendLo) / blendRange));

    hwc[idx]     = Math.max(0, fr + alpha * (r - fr));
    hwc[idx + 1] = Math.max(0, fg + alpha * (g - fg));
    hwc[idx + 2] = Math.max(0, fb + alpha * (b - fb));
  }
}

export function applyExifRotation(
  hwc: Float32Array, width: number, height: number, orientation: string,
): RotatedImage {
  if (orientation === 'Normal' || orientation === 'Unknown') {
    return { data: hwc, width, height };
  }

  const n = width * height;

  if (orientation === 'Rotate180') {
    const out = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const si = (n - 1 - i) * 3;
      const di = i * 3;
      out[di] = hwc[si]; out[di + 1] = hwc[si + 1]; out[di + 2] = hwc[si + 2];
    }
    return { data: out, width, height };
  }

  if (orientation === 'Rotate90' || orientation === 'Rotate270') {
    const out = new Float32Array(n * 3);
    const newW = height, newH = width;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const si = (y * width + x) * 3;
        let dx: number, dy: number;
        if (orientation === 'Rotate90') {
          dx = height - 1 - y;
          dy = x;
        } else {
          dx = y;
          dy = width - 1 - x;
        }
        const di = (dy * newW + dx) * 3;
        out[di] = hwc[si]; out[di + 1] = hwc[si + 1]; out[di + 2] = hwc[si + 2];
      }
    }
    return { data: out, width: newW, height: newH };
  }

  return { data: hwc, width, height };
}

function linearToSrgb8(v: number): number {
  v = Math.max(0, Math.min(1, v));
  if (v <= 0.0031308) {
    return (v * 12.92 * 255 + 0.5) | 0;
  }
  return ((1.055 * Math.pow(v, 1 / 2.4) - 0.055) * 255 + 0.5) | 0;
}

export function toImageData(hwc: Float32Array, width: number, height: number): ImageData {
  const n = width * height;
  const rgba = new Uint8ClampedArray(n * 4);

  for (let i = 0; i < n; i++) {
    const si = i * 3;
    const di = i * 4;
    rgba[di]     = linearToSrgb8(hwc[si]);
    rgba[di + 1] = linearToSrgb8(hwc[si + 1]);
    rgba[di + 2] = linearToSrgb8(hwc[si + 2]);
    rgba[di + 3] = 255;
  }

  return new ImageData(rgba, width, height);
}
