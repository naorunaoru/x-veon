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

/**
 * Inpaint-opposed highlight reconstruction (adapted from darktable).
 * For clipped CFA pixels, estimates the true value from the opposed-channel
 * reference average. Per-channel means are computed in linear space, converted
 * to cube-root space, then the two opposing channel means are averaged.
 * Chrominance correction is computed and applied in linear space.
 * Operates before WB so clipped channels can be extended above the clip
 * point, preventing channel imbalance after WB multiplication.
 */
export function reconstructHighlightsCfa(
  cfa: Float32Array, width: number, height: number,
  pattern: readonly (readonly number[])[], period: number,
  dy: number, dx: number,
): void {
  const clip = 1.0;
  const chromLo = 0.2;

  function getCh(y: number, x: number): number {
    return pattern[((y + dy) % period + period) % period][((x + dx) % period + period) % period];
  }

  // Compute opposed-channel reference average for pixel (y, x) with channel ch.
  // Accumulates per-channel means in linear space within a 3x3 neighborhood,
  // converts to cube-root space, averages the two opposing channels, returns linear.
  function calcRefavg(y: number, x: number, ch: number): number {
    const mean = [0, 0, 0];
    const cnt = [0, 0, 0];

    const y0 = Math.max(0, y - 1);
    const y1 = Math.min(height - 1, y + 1);
    const x0 = Math.max(0, x - 1);
    const x1 = Math.min(width - 1, x + 1);

    for (let ny = y0; ny <= y1; ny++) {
      for (let nx = x0; nx <= x1; nx++) {
        const val = Math.max(0, cfa[ny * width + nx]);
        const c = getCh(ny, nx);
        mean[c] += val;
        cnt[c] += 1;
      }
    }

    // Per-channel mean in linear space, then convert to cube-root
    const cr = [0, 0, 0];
    for (let c = 0; c < 3; c++) {
      cr[c] = cnt[c] > 0 ? Math.cbrt(mean[c] / cnt[c]) : 0;
    }

    // Opposed = average of the other two channels in cube-root space
    let oppCr: number;
    if (ch === 0) oppCr = 0.5 * (cr[1] + cr[2]);
    else if (ch === 1) oppCr = 0.5 * (cr[0] + cr[2]);
    else oppCr = 0.5 * (cr[0] + cr[1]);

    return oppCr * oppCr * oppCr;
  }

  // Pass 1: global chrominance correction from near-clip unclipped pixels.
  // Chrominance = average of (linear_value - refavg) in linear space.
  const chromSum = [0, 0, 0];
  const chromCnt = [0, 0, 0];

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const val = cfa[y * width + x];
      if (val < chromLo || val >= clip) continue;

      const ch = getCh(y, x);
      const ref = calcRefavg(y, x, ch);
      chromSum[ch] += val - ref;
      chromCnt[ch]++;
    }
  }

  const chrom = [0, 0, 0];
  for (let c = 0; c < 3; c++) {
    if (chromCnt[c] > 100) chrom[c] = chromSum[c] / chromCnt[c];
  }

  // Pass 2: for clipped pixels, estimate from opposed-channel reference average.
  // Reconstruction in linear space: max(clipped_value, refavg + chrominance).
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = y * width + x;
      if (cfa[idx] < clip) continue;

      const ch = getCh(y, x);
      const ref = calcRefavg(y, x, ch);
      const estimate = ref + chrom[ch];
      cfa[idx] = Math.max(cfa[idx], estimate);
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
