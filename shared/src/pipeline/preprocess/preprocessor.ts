import { XTRANS_PATTERN, BAYER_PATTERN } from '../constants';
import type { CfaInfo, PaddedCfa, PreparedCfa, RawImage, Rect, TileGrid, ChannelMasks } from '../types';

/** Clip threshold as a fraction of the calibrated white level. */
const CLIP_MAGIC = 0.96;

/** The visible area of a sensor readout, from rawloader's crops (top, right, bottom, left). */
export function visibleRect(fullWidth: number, fullHeight: number, crops: Uint16Array): Rect {
  const top = crops[0], right = crops[1], bottom = crops[2], left = crops[3];
  return { left, top, width: fullWidth - left - right, height: fullHeight - top - bottom };
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
  rawData: Uint16Array, stride: number, rect: Rect,
  whiteLevels: Uint16Array, cfa: Pick<CfaInfo, 'pattern' | 'period' | 'dy' | 'dx'>,
): Uint16Array {
  const lut = visibleColorLut(cfa);
  const period = cfa.period;
  const { left, top, width, height } = rect;
  // One pass: per colour, the maximum, how many photosites sit at it, and how many sit one DN
  // below it (together: the count within 1 DN of the maximum, as the old second pass counted).
  const max = [0, 0, 0];
  const atMax = [0, 0, 0];
  const belowMax = [0, 0, 0];
  const cnt = [0, 0, 0];

  for (let y = 0; y < height; y++) {
    const row = (y + top) * stride + left;
    const lutRow = (y % period) * period;
    for (let x = 0, px = 0; x < width; x++, px = px + 1 === period ? 0 : px + 1) {
      const c = lut[lutRow + px];
      const v = rawData[row + x];
      cnt[c]++;
      const m = max[c];
      if (v === m) atMax[c]++;
      else if (v > m) {
        belowMax[c] = v === m + 1 ? atMax[c] : 0;
        atMax[c] = 1;
        max[c] = v;
      } else if (v === m - 1) belowMax[c]++;
    }
  }

  const calibrated = new Uint16Array(4);
  for (let c = 0; c < 4; c++) {
    const wl = levelFor(whiteLevels, c);
    // Clipping detected AND actual saturation is below metadata white level. With a maximum
    // of 0 every photosite is within 1 DN of it, as before.
    const nearMax = c < 3 ? (max[c] === 0 ? cnt[c] : atMax[c] + belowMax[c]) : 0;
    calibrated[c] = c < 3 && nearMax > cnt[c] * 1e-4 && max[c] < wl ? max[c] : wl;
  }
  return calibrated;
}

/** Values per colour in a normalisation table: every u16 raw value. */
export const NORM_LUT_SIZE = 65536;

/**
 * Per-colour black/white normalisation as a table, `lut[colour * NORM_LUT_SIZE + value]`, so
 * every channel clips at exactly 1.0. Levels are indexed by the photosite's CFA colour:
 * rawloader reports them in RGBE order (for Canon, E is 0 because it averages masked areas per
 * colour). The table holds the same float32 values the per-pixel division produced, and lets the
 * CFA stay u16 until it is consumed (on the GPU, or per strip in the WASM pool).
 */
export function normalizationLut(blackLevels: Uint16Array, whiteLevels: Uint16Array): Float32Array {
  const lut = new Float32Array(3 * NORM_LUT_SIZE);
  for (let c = 0; c < 3; c++) {
    const sub = levelFor(blackLevels, c);
    const div = levelFor(whiteLevels, c) - sub;
    const base = c * NORM_LUT_SIZE;
    for (let v = 0; v < NORM_LUT_SIZE; v++) lut[base + v] = (v - sub) / div;
  }
  return lut;
}

/**
 * Normalise rows [rowStart, rowEnd) of a canonically aligned u16 CFA to float32.
 * `pattern` is the canonical period × period colour layout, row-major.
 */
export function normalizeRows(
  cfa: Uint16Array, width: number, rowStart: number, rowEnd: number,
  pattern: Uint32Array, period: number, lut: Float32Array,
): Float32Array {
  const out = new Float32Array((rowEnd - rowStart) * width);
  const base = new Uint32Array(period * period);
  for (let i = 0; i < base.length; i++) base[i] = pattern[i] * NORM_LUT_SIZE;
  for (let y = rowStart; y < rowEnd; y++) {
    const src = y * width;
    const dst = (y - rowStart) * width;
    const patRow = (y % period) * period;
    for (let x = 0, px = 0; x < width; x++, px = px + 1 === period ? 0 : px + 1) {
      out[dst + x] = lut[base[patRow + px] + cfa[src + x]];
    }
  }
  return out;
}

/**
 * Crop the visible area out of the readout and pad its top and left so the CFA phase matches the
 * reference pattern, in one copy. Pad rows and columns repeat the nearest source rows/columns
 * with the same CFA phase (mirroring would put values of the wrong colour next to the edge, where
 * the demosaic reads them as context). With nothing to crop or pad, the readout itself is returned.
 */
export function cropAndPad(
  rawData: Uint16Array, stride: number, rect: Rect, dy: number, dx: number, period: number,
): PaddedCfa {
  const { left, top, width, height } = rect;
  const padTop = dy;
  const padLeft = dx;
  const newW = width + padLeft;
  const newH = height + padTop;

  if (padTop === 0 && padLeft === 0 && left === 0 && top === 0 && width === stride
      && height * stride === rawData.length) {
    return { data: rawData, width, height, padTop: 0, padLeft: 0 };
  }

  const samePhase = (i: number, pad: number, size: number) =>
    Math.min(i < pad ? (((i - pad) % period) + period) % period : i - pad, size - 1);

  const out = new Uint16Array(newW * newH);
  for (let y = 0; y < newH; y++) {
    const src = (samePhase(y, padTop, height) + top) * stride + left;
    const dst = y * newW;
    for (let x = 0; x < padLeft; x++) out[dst + x] = rawData[src + samePhase(x, padLeft, width)];
    out.set(rawData.subarray(src, src + width), dst + padLeft);
  }

  return { data: out, width: newW, height: newH, padTop, padLeft };
}

/**
 * Everything the demosaic needs from a decoded readout, computed where it was decoded (the
 * decode worker): the CFA layout, calibrated white levels, the cropped and phase-aligned u16 CFA
 * and its normalisation table.
 */
export function prepareCfa(raw: RawImage): PreparedCfa {
  const rect = visibleRect(raw.width, raw.height, raw.crops);
  const cfa = findPatternShift(raw.cfaStr, raw.cfaWidth, raw.crops);
  const whiteLevels = calibrateWhiteLevels(raw.data, raw.width, rect, raw.whiteLevels, cfa);
  const padded = cropAndPad(raw.data, raw.width, rect, cfa.dy, cfa.dx, cfa.period);
  return {
    ...padded,
    visibleWidth: rect.width,
    visibleHeight: rect.height,
    cfa,
    whiteLevels,
    lut: normalizationLut(raw.blackLevels, whiteLevels),
  };
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
