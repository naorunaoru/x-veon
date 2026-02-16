import { XYZ_TO_SRGB } from './constants';

export interface TileBlender {
  accumulate(tile: Float32Array, tx: number, ty: number): void;
  finalize(): Float32Array;
}

export function createTileBlender(
  hPad: number, wPad: number, patchSize: number, overlap: number,
): TileBlender {
  const planeSize = hPad * wPad;
  const output = new Float32Array(3 * planeSize);
  const weights = new Float32Array(planeSize);
  const patchPixels = patchSize * patchSize;

  const w1d = new Float32Array(patchSize);
  w1d.fill(1);
  for (let i = 0; i < overlap; i++) {
    w1d[i] = i / overlap;
    w1d[patchSize - 1 - i] = i / overlap;
  }

  return {
    accumulate(tile: Float32Array, tx: number, ty: number): void {
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
    },

    finalize(): Float32Array {
      for (let i = 0; i < planeSize; i++) {
        if (weights[i] > 1e-8) {
          const invW = 1 / weights[i];
          for (let c = 0; c < 3; c++) {
            output[c * planeSize + i] *= invW;
          }
        }
      }
      return output;
    },
  };
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
  for (let i = 0; i < numPixels; i++) {
    const idx = i * 3;
    const r = hwc[idx], g = hwc[idx + 1], b = hwc[idx + 2];

    hwc[idx]     = Math.max(0, matrix[0] * r + matrix[1] * g + matrix[2] * b);
    hwc[idx + 1] = Math.max(0, matrix[3] * r + matrix[4] * g + matrix[5] * b);
    hwc[idx + 2] = Math.max(0, matrix[6] * r + matrix[7] * g + matrix[8] * b);
  }
}

