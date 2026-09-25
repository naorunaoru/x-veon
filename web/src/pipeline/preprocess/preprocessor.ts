import { XTRANS_PATTERN, BAYER_PATTERN } from '../constants';
import type { CfaInfo, CroppedImage, PaddedImage, TileGrid, ChannelMasks } from '../types';

/** Clip threshold as a fraction of the calibrated white level. */
const CLIP_MAGIC = 0.96;

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

/**
 * Per-channel normalized clip levels.
 * With per-colour normalization in normalizeRawCfa, every channel
 * clips at exactly 1.0, so clips are simply CLIP_MAGIC for all channels.
 */
export function channelClips(): [number, number, number] {
  return [CLIP_MAGIC, CLIP_MAGIC, CLIP_MAGIC];
}

/** The CFA colour (0=R, 1=G, 2=B) of every position in one period of the visible image, row-major. */
export function visibleColorLut(cfa: Pick<CfaInfo, 'pattern' | 'period' | 'dy' | 'dx'>): Uint8Array {
  const { pattern, period, dy, dx } = cfa;
  const lut = new Uint8Array(period * period);
  for (let y = 0; y < period; y++) {
    for (let x = 0; x < period; x++) {
      lut[y * period + x] = pattern[(y + dy) % period][(x + dx) % period];
    }
  }
  return lut;
}

/** Level for a CFA colour. rawloader reports levels in RGBE order, not by photosite position. */
function levelFor(levels: Uint16Array, color: number): number {
  return color < levels.length ? levels[color] : levels[0];
}

/**
 * White-point calibration: detect actual sensor saturation from raw data.
 *
 * Camera databases (rawloader TOML, etc.) often report the theoretical ADC
 * maximum (e.g. 16383 for 14-bit) rather than the real photosite saturation
 * which can be significantly lower.  RawSpeed's cameras.xml tends to have
 * empirically calibrated values, so darktable doesn't hit this problem.
 *
 * For each CFA colour we find the actual data maximum.  If a meaningful
 * number of pixels sit at that maximum (≥ 0.01 % of that colour's pixel
 * count) — indicating real sensor clipping — AND the maximum is below the
 * metadata white level, we adopt the measured value as the effective white
 * point.  Returns levels in RGBE order, like the input.
 */
export function calibrateWhiteLevels(
  rawData: Uint16Array, width: number, height: number,
  whiteLevels: Uint16Array, cfa: Pick<CfaInfo, 'pattern' | 'period' | 'dy' | 'dx'>,
): Uint16Array {
  const lut = visibleColorLut(cfa);
  const period = cfa.period;
  const max = [0, 0, 0];
  const cnt = [0, 0, 0];

  for (let y = 0; y < height; y++) {
    const row = y * width;
    const lutRow = (y % period) * period;
    for (let x = 0, px = 0; x < width; x++, px = px + 1 === period ? 0 : px + 1) {
      const c = lut[lutRow + px];
      const v = rawData[row + x];
      if (v > max[c]) max[c] = v;
      cnt[c]++;
    }
  }

  // Second pass: count pixels at the detected maximum (within 1 DN)
  const atMax = [0, 0, 0];
  for (let y = 0; y < height; y++) {
    const row = y * width;
    const lutRow = (y % period) * period;
    for (let x = 0, px = 0; x < width; x++, px = px + 1 === period ? 0 : px + 1) {
      const c = lut[lutRow + px];
      if (rawData[row + x] >= max[c] - 1) atMax[c]++;
    }
  }

  const calibrated = new Uint16Array(4);
  for (let c = 0; c < 4; c++) {
    const wl = levelFor(whiteLevels, c);
    // Clipping detected AND actual saturation is below metadata white level
    calibrated[c] = c < 3 && atMax[c] > cnt[c] * 1e-4 && max[c] < wl ? max[c] : wl;
  }
  return calibrated;
}

export function normalizeRawCfa(
  rawData: Uint16Array, width: number, height: number,
  blackLevels: Uint16Array, whiteLevels: Uint16Array,
  cfa: Pick<CfaInfo, 'pattern' | 'period' | 'dy' | 'dx'>,
): Float32Array {
  // Per-colour black/white calibration, so every channel clips at exactly 1.0 after
  // normalization. Levels are indexed by the photosite's CFA colour: rawloader reports
  // them in RGBE order (for Canon, E is 0 because it averages masked areas per colour).
  const sub = new Float32Array(3);
  const div = new Float32Array(3);
  for (let c = 0; c < 3; c++) {
    sub[c] = levelFor(blackLevels, c);
    div[c] = levelFor(whiteLevels, c) - sub[c];
  }

  const lut = visibleColorLut(cfa);
  const period = cfa.period;
  const n = width * height;
  const out = new Float32Array(n);
  for (let y = 0; y < height; y++) {
    const row = y * width;
    const lutRow = (y % period) * period;
    for (let x = 0, px = 0; x < width; x++, px = px + 1 === period ? 0 : px + 1) {
      const c = lut[lutRow + px];
      out[row + x] = (rawData[row + x] - sub[c]) / div[c];
    }
  }
  return out;
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
  width: number, height: number, patchSize: number, overlap: number,
): TileGrid {
  const stride = patchSize - overlap;
  const hPad = Math.ceil((height - overlap) / stride) * stride + patchSize;
  const wPad = Math.ceil((width - overlap) / stride) * stride + patchSize;

  const tiles: Array<{ x: number; y: number }> = [];
  for (let y = 0; y <= hPad - patchSize; y += stride) {
    for (let x = 0; x <= wPad - patchSize; x += stride) {
      tiles.push({ x, y });
    }
  }

  return { tiles, hPad, wPad };
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
