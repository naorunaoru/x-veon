import { XTRANS_PATTERN, BAYER_PATTERN } from './constants';
import type { CfaInfo, CroppedImage, PaddedImage, TileGrid, ChannelMasks } from './types';

export function cropToVisible(
  rawData: Uint16Array, fullWidth: number, fullHeight: number, crops: Uint16Array,
): CroppedImage {
  const top = crops[0], right = crops[1], bottom = crops[2], left = crops[3];
  const visW = fullWidth - left - right;
  const visH = fullHeight - top - bottom;

  if (top === 0 && right === 0 && bottom === 0 && left === 0) {
    return { data: rawData, width: fullWidth, height: fullHeight };
  }

  const out = new Uint16Array(visW * visH);
  for (let y = 0; y < visH; y++) {
    const srcOffset = (y + top) * fullWidth + left;
    out.set(rawData.subarray(srcOffset, srcOffset + visW), y * visW);
  }

  return { data: out, width: visW, height: visH };
}

function parseCfaStr(cfaStr: string, cfaWidth: number): number[][] {
  const cfaHeight = cfaStr.length / cfaWidth;
  const pattern: number[][] = [];
  for (let y = 0; y < cfaHeight; y++) {
    pattern[y] = [];
    for (let x = 0; x < cfaWidth; x++) {
      const ch = cfaStr[y * cfaWidth + x];
      pattern[y][x] = ch === 'R' ? 0 : ch === 'G' ? 1 : 2;
    }
  }
  return pattern;
}

function matchShift(
  canonical: readonly (readonly number[])[],
  visible: number[][],
  period: number,
): { dy: number; dx: number } | null {
  for (let dy = 0; dy < period; dy++) {
    for (let dx = 0; dx < period; dx++) {
      let match = true;
      for (let y = 0; y < period && match; y++) {
        for (let x = 0; x < period && match; x++) {
          if (canonical[(y + dy) % period][(x + dx) % period] !== visible[y][x]) {
            match = false;
          }
        }
      }
      if (match) return { dy, dx };
    }
  }
  return null;
}

export function findPatternShift(cfaStr: string, cfaWidth: number, crops: Uint16Array): CfaInfo {
  const period = cfaWidth;
  const rawPattern = parseCfaStr(cfaStr, cfaWidth);

  // Apply crop offset to get the visible CFA pattern
  const top = crops[0], left = crops[3];
  const vis: number[][] = [];
  for (let y = 0; y < period; y++) {
    vis[y] = [];
    for (let x = 0; x < period; x++) {
      vis[y][x] = rawPattern[(y + top) % period][(x + left) % period];
    }
  }

  if (period === 6) {
    const shift = matchShift(XTRANS_PATTERN, vis, 6);
    if (shift) {
      return { cfaType: 'xtrans', pattern: XTRANS_PATTERN, period: 6, ...shift };
    }
    throw new Error(`CFA pattern does not match X-Trans reference: ${cfaStr}`);
  }

  if (period === 2) {
    const shift = matchShift(BAYER_PATTERN, vis, 2);
    if (shift) {
      return { cfaType: 'bayer', pattern: BAYER_PATTERN, period: 2, ...shift };
    }
    throw new Error(`CFA pattern does not match Bayer reference: ${cfaStr}`);
  }

  throw new Error(`Unsupported CFA: ${cfaStr.length} chars, width ${cfaWidth}`);
}

export function normalizeRawCfa(
  rawData: Uint16Array, width: number, height: number,
  blackLevels: Uint16Array, whiteLevels: Uint16Array,
): Float32Array {
  const black = blackLevels[0];
  const white = whiteLevels[0];
  const range = white - black;
  const n = width * height;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = (rawData[i] - black) / range;
  }
  return out;
}

export function reconstructHighlightsCfa(
  cfa: Float32Array, width: number, height: number,
  pattern: readonly (readonly number[])[], period: number,
  dy: number, dx: number,
): void {
  const SQRT3 = Math.sqrt(3);
  const SQRT12 = 2 * SQRT3;
  const clip = 1.0;

  const src = new Float32Array(cfa);

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      let anyClipped = false;
      for (let jj = -1; jj <= 1 && !anyClipped; jj++) {
        const row = (y + jj) * width;
        for (let ii = -1; ii <= 1 && !anyClipped; ii++) {
          if (src[row + x + ii] >= clip) anyClipped = true;
        }
      }
      if (!anyClipped) continue;

      const chMax = [-Infinity, -Infinity, -Infinity];
      const chSum = [0, 0, 0];
      const chCnt = [0, 0, 0];

      for (let jj = -1; jj <= 1; jj++) {
        const row = (y + jj) * width;
        for (let ii = -1; ii <= 1; ii++) {
          const val = src[row + x + ii];
          const ch = pattern[((y + jj + dy) % period + period) % period][((x + ii + dx) % period + period) % period];
          if (val > chMax[ch]) chMax[ch] = val;
          chSum[ch] += Math.min(val, clip);
          chCnt[ch]++;
        }
      }

      const R = chMax[0], G = chMax[1], B = chMax[2];
      const Ro = Math.min(chSum[0] / chCnt[0], clip);
      const Go = Math.min(chSum[1] / chCnt[1], clip);
      const Bo = Math.min(chSum[2] / chCnt[2], clip);

      const L = (R + G + B) / 3;
      let C = SQRT3 * (R - G);
      let H = 2 * B - G - R;
      const Co = SQRT3 * (Ro - Go);
      const Ho = 2 * Bo - Go - Ro;

      const denom = C * C + H * H;
      if (R !== G && G !== B && denom > 1e-20) {
        const ratio = Math.sqrt((Co * Co + Ho * Ho) / denom);
        C *= ratio;
        H *= ratio;
      }

      const RGB = [
        L - H / 6 + C / SQRT12,
        L - H / 6 - C / SQRT12,
        L + H / 3,
      ];

      const ch = pattern[((y + dy) % period + period) % period][((x + dx) % period + period) % period];
      cfa[y * width + x] = RGB[ch];
    }
  }
}

export function applyWhiteBalance(
  cfa: Float32Array, width: number, height: number,
  wb: Float32Array,
  pattern: readonly (readonly number[])[], period: number,
  dy: number, dx: number,
): void {
  for (let y = 0; y < height; y++) {
    const patY = (y + dy) % period;
    const row = y * width;
    for (let x = 0; x < width; x++) {
      const ch = pattern[patY][(x + dx) % period];
      cfa[row + x] *= wb[ch];
    }
  }
}

export function padToAlignment(
  cfa: Float32Array, width: number, height: number, dy: number, dx: number,
): PaddedImage {
  const padTop = dy;
  const padLeft = dx;

  if (padTop === 0 && padLeft === 0) {
    return { data: cfa, width, height, padTop: 0, padLeft: 0 };
  }

  const newW = width + padLeft;
  const newH = height + padTop;
  const out = new Float32Array(newW * newH);

  for (let y = 0; y < newH; y++) {
    const srcY = y < padTop ? padTop - 1 - y : y - padTop;
    const clampedSrcY = Math.min(srcY, height - 1);
    for (let x = 0; x < newW; x++) {
      const srcX = x < padLeft ? padLeft - 1 - x : x - padLeft;
      const clampedSrcX = Math.min(srcX, width - 1);
      out[y * newW + x] = cfa[clampedSrcY * width + clampedSrcX];
    }
  }

  return { data: out, width: newW, height: newH, padTop, padLeft };
}

export function generateTiles(
  cfa: Float32Array, width: number, height: number, patchSize: number, overlap: number,
): TileGrid {
  const stride = patchSize - overlap;
  const hPad = Math.ceil((height - overlap) / stride) * stride + patchSize;
  const wPad = Math.ceil((width - overlap) / stride) * stride + patchSize;

  const paddedCfa = new Float32Array(hPad * wPad);
  for (let y = 0; y < height; y++) {
    paddedCfa.set(cfa.subarray(y * width, y * width + width), y * wPad);
  }

  const tiles: Array<{ x: number; y: number }> = [];
  for (let y = 0; y <= hPad - patchSize; y += stride) {
    for (let x = 0; x <= wPad - patchSize; x += stride) {
      tiles.push({ x, y });
    }
  }

  return { tiles, paddedCfa, hPad, wPad };
}

export function makeChannelMasks(
  patchSize: number,
  pattern: readonly (readonly number[])[],
  period: number,
): ChannelMasks {
  const n = patchSize * patchSize;
  const r = new Float32Array(n);
  const g = new Float32Array(n);
  const b = new Float32Array(n);
  for (let y = 0; y < patchSize; y++) {
    const patY = y % period;
    const row = y * patchSize;
    for (let x = 0; x < patchSize; x++) {
      const ch = pattern[patY][x % period];
      const idx = row + x;
      if (ch === 0) r[idx] = 1;
      else if (ch === 1) g[idx] = 1;
      else b[idx] = 1;
    }
  }
  return { r, g, b };
}

export function buildTileInput(
  paddedCfa: Float32Array, paddedWidth: number,
  x: number, y: number, patchSize: number, masks: ChannelMasks,
): Float32Array {
  const n = patchSize * patchSize;
  const input = new Float32Array(4 * n);

  for (let py = 0; py < patchSize; py++) {
    const srcOffset = (y + py) * paddedWidth + x;
    const dstOffset = py * patchSize;
    input.set(paddedCfa.subarray(srcOffset, srcOffset + patchSize), dstOffset);
  }

  input.set(masks.r, n);
  input.set(masks.g, 2 * n);
  input.set(masks.b, 3 * n);

  return input;
}
