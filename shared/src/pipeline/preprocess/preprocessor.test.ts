import { describe, it, expect } from 'vitest';
import {
  visibleRect, findPatternShift, channelClips, calibrateWhiteLevels, normalizationLut, normalizeRows,
  cropAndPad, prepareCfa, NORM_LUT_SIZE, generateTiles, makeChannelMasks, visibleColorLut,
} from './preprocessor';
import type { CfaInfo, RawImage } from '../types';
import { XTRANS_PATTERN, BAYER_PATTERN } from '../constants';

function cfaString(pattern: readonly (readonly number[])[], period: number, dy: number, dx: number): string {
  let s = '';
  for (let y = 0; y < period; y++) {
    for (let x = 0; x < period; x++) s += 'RGB'[pattern[(y + dy) % period][(x + dx) % period]];
  }
  return s;
}

describe('visibleRect', () => {
  it('turns top/right/bottom/left crops into a rectangle', () => {
    expect(visibleRect(10, 8, new Uint16Array([1, 2, 3, 4]))).toEqual({ left: 4, top: 1, width: 4, height: 4 });
  });
});

describe('findPatternShift', () => {
  it('recognises the X-Trans reference with a shift and applies the crop offset', () => {
    const str = cfaString(XTRANS_PATTERN, 6, 2, 3);
    const noCrop = findPatternShift(str, 6, new Uint16Array([0, 0, 0, 0]));
    expect(noCrop).toMatchObject({ cfaType: 'xtrans', period: 6, dy: 2, dx: 3 });
    // The visible pattern is now shifted by (3,5). The X-Trans reference maps onto itself under a
    // (3,3) shift, so (0,2) also matches, and matchShift returns the first hit in scan order.
    const cropped = findPatternShift(str, 6, new Uint16Array([1, 0, 0, 2]));
    expect(cropped).toMatchObject({ dy: 0, dx: 2 });
  });
  it('recognises Bayer variants', () => {
    expect(findPatternShift('GRBG', 2, new Uint16Array(4))).toMatchObject({ cfaType: 'bayer', dy: 0, dx: 1 });
    expect(findPatternShift('BGGR', 2, new Uint16Array(4))).toMatchObject({ dy: 1, dx: 1 });
  });
  it('rejects patterns that match no reference', () => {
    expect(() => findPatternShift('RRRR', 2, new Uint16Array(4))).toThrow(/Bayer/);
    expect(() => findPatternShift('RGBRGBRGB', 3, new Uint16Array(4))).toThrow(/Unsupported CFA/);
  });
});

describe('channelClips', () => {
  it('is the same threshold for every channel', () => {
    expect(channelClips()).toEqual([0.96, 0.96, 0.96]);
  });
});

const RGGB = { pattern: [[0, 1], [1, 2]], period: 2, dy: 0, dx: 0 } as const;

describe('visibleColorLut', () => {
  it('maps every visible position of one period to its CFA colour', () => {
    expect(Array.from(visibleColorLut(RGGB))).toEqual([0, 1, 1, 2]);
    // GBRG as seen by findPatternShift: RGGB reference shifted by one row.
    expect(Array.from(visibleColorLut(findPatternShift('GBRG', 2, new Uint16Array(4))))).toEqual([1, 2, 0, 1]);
  });
});

const full = (w: number, h: number) => ({ left: 0, top: 0, width: w, height: h });

/** The previous two-pass calibration, kept as the reference the single pass must match. */
function calibrateReference(
  data: Uint16Array, w: number, h: number, whites: Uint16Array, cfa: Pick<CfaInfo, 'pattern' | 'period' | 'dy' | 'dx'>,
): number[] {
  const lut = visibleColorLut(cfa);
  const p = cfa.period;
  const max = [0, 0, 0], cnt = [0, 0, 0], atMax = [0, 0, 0];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const c = lut[(y % p) * p + (x % p)];
    max[c] = Math.max(max[c], data[y * w + x]); cnt[c]++;
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const c = lut[(y % p) * p + (x % p)];
    if (data[y * w + x] >= max[c] - 1) atMax[c]++;
  }
  return [0, 1, 2, 3].map((c) => {
    const wl = c < whites.length ? whites[c] : whites[0];
    return c < 3 && atMax[c] > cnt[c] * 1e-4 && max[c] < wl ? max[c] : wl;
  });
}

/** Deterministic pseudo-random u16 data clustered near `top`, so maxima and near-maxima repeat. */
function noisy(n: number, top: number, seed: number): Uint16Array {
  const out = new Uint16Array(n);
  let x = seed;
  for (let i = 0; i < n; i++) {
    x = (x * 1103515245 + 12345) >>> 0;
    out[i] = Math.max(0, top - ((x >>> 16) % 5) * ((x >>> 8) % 3));
  }
  return out;
}

describe('calibrateWhiteLevels', () => {
  it('adopts a measured saturation that sits below the metadata white level', () => {
    // 200×200 RGGB: red saturates at 4000 on 10% of its pixels; green and blue each have a single
    // bright outlier (not above the 0.01% threshold), so they keep the metadata level.
    const w = 200, h = 200;
    const data = new Uint16Array(w * h).fill(100);
    data[1] = 500; data[w + 1] = 500;
    for (let y = 0; y < h; y += 2) for (let x = 0; x < w; x += 20) data[y * w + x] = 4000;
    const out = calibrateWhiteLevels(data, w, full(w, h), new Uint16Array([16383, 16383, 16383, 16383]), RGGB);
    expect(Array.from(out)).toEqual([4000, 16383, 16383, 16383]);
  });
  it('keeps the metadata level when the measured maximum is at or above it', () => {
    const data = new Uint16Array(16).fill(1000);
    data[0] = 1200;
    expect(Array.from(calibrateWhiteLevels(data, 4, full(4, 4), new Uint16Array([1000]), RGGB))).toEqual([1000, 1000, 1000, 1000]);
  });
  it('matches the two-pass count within 1 DN of the maximum, including maxima of 0 and 1', () => {
    const cfa = findPatternShift(cfaString(XTRANS_PATTERN, 6, 2, 3), 6, new Uint16Array(4));
    for (const [top, seed] of [[4000, 1], [4000, 7], [1, 3], [0, 5], [2, 9]]) {
      const data = noisy(60 * 48, top, seed);
      const whites = new Uint16Array([16383, 16383, 16383, 16383]);
      expect(Array.from(calibrateWhiteLevels(data, 60, full(60, 48), whites, cfa)))
        .toEqual(calibrateReference(data, 60, 48, whites, cfa));
      // A 1-DN step below 0.01% of the pixels must not count; above it must.
      const lifted = data.slice(); lifted[0] = top + 1;
      expect(Array.from(calibrateWhiteLevels(lifted, 60, full(60, 48), whites, cfa)))
        .toEqual(calibrateReference(lifted, 60, 48, whites, cfa));
    }
  });
  it('measures only the visible rectangle of the readout', () => {
    const data = new Uint16Array(8 * 6).fill(100);
    data[0] = 60000;  // outside the rectangle
    const out = calibrateWhiteLevels(data, 8, { left: 2, top: 2, width: 4, height: 4 }, new Uint16Array([16383]), RGGB);
    expect(Array.from(out)).toEqual([100, 100, 100, 16383]);
  });
});

describe('normalizationLut / normalizeRows', () => {
  const PAT_RGGB = new Uint32Array([0, 1, 1, 2]);
  it('normalises each photosite with the black and white level of its colour', () => {
    // RGGB: R, G, G, B. Levels are in RGBE order, as rawloader reports them.
    const lut = normalizationLut(new Uint16Array([10, 20, 30, 40]), new Uint16Array([210, 220, 230, 240]));
    const out = normalizeRows(new Uint16Array([110, 220, 220, 330]), 2, 0, 2, PAT_RGGB, 2, lut);
    expect(Array.from(out)).toEqual([0.5, 1.0, 1.0, 1.5]);
  });
  it('ignores the unused E slot (a Canon black level of 0 there must not reach blue)', () => {
    // Canon masked-area blacks: [R, G, B, E=0]. A black frame must normalise to 0 everywhere.
    const lut = normalizationLut(new Uint16Array([1024, 1027, 1030, 0]), new Uint16Array([15600, 15600, 15600, 15600]));
    expect(Array.from(normalizeRows(new Uint16Array([1024, 1027, 1027, 1030]), 2, 0, 2, PAT_RGGB, 2, lut))).toEqual([0, 0, 0, 0]);
  });
  it('holds exactly the float32 values of the per-pixel division', () => {
    const blacks = new Uint16Array([512, 510, 515, 0]);
    const whites = new Uint16Array([15000, 16383, 14000, 0]);
    const lut = normalizationLut(blacks, whites);
    for (let c = 0; c < 3; c++) {
      for (const v of [0, 1, 511, 512, 513, 9999, 14000, 16383, 65535]) {
        expect(lut[c * NORM_LUT_SIZE + v]).toBe(Math.fround((v - blacks[c]) / (whites[c] - blacks[c])));
      }
    }
  });
  it('follows the X-Trans colour layout, and normalises only the rows asked for', () => {
    const pattern = new Uint32Array(XTRANS_PATTERN.flat());
    const data = new Uint16Array(12 * 6);
    for (let i = 0; i < data.length; i++) data[i] = [100, 200, 300][pattern[((Math.floor(i / 6) % 6) * 6) + (i % 6)]];
    const lut = normalizationLut(new Uint16Array([100, 200, 300, 0]), new Uint16Array([1100, 1200, 1300, 0]));
    const out = normalizeRows(data, 6, 3, 9, pattern, 6, lut);
    expect(out.length).toBe(36);
    expect(Array.from(out).every((v) => v === 0)).toBe(true);
  });
});

describe('cropAndPad', () => {
  it('returns the readout itself with nothing to crop or pad', () => {
    const data = new Uint16Array([1, 2, 3, 4]);
    expect(cropAndPad(data, 2, full(2, 2), 0, 0, 2).data).toBe(data);
  });
  it('crops top/right/bottom/left', () => {
    const data = new Uint16Array([...Array(16).keys()]);  // 4×4
    const out = cropAndPad(data, 4, visibleRect(4, 4, new Uint16Array([1, 1, 1, 1])), 0, 0, 2);
    expect([out.width, out.height]).toEqual([2, 2]);
    expect(Array.from(out.data)).toEqual([5, 6, 9, 10]);
  });
  it('pads with source rows and columns of the same CFA phase', () => {
    const data = new Uint16Array([1, 2, 3, 4, 5, 6]);  // 3×2
    const out = cropAndPad(data, 3, full(3, 2), 1, 1, 2);
    expect([out.width, out.height, out.padTop, out.padLeft]).toEqual([4, 3, 1, 1]);
    // Pad row 0 has the phase of source row 1, pad column 0 that of source column 1.
    expect(Array.from(out.data)).toEqual([
      5, 4, 5, 6,
      2, 1, 2, 3,
      5, 4, 5, 6,
    ]);
  });
  it('keeps every padded X-Trans photosite on the reference colour, after a crop', () => {
    const w = 16, h = 15, crops = new Uint16Array([1, 2, 2, 2]);  // top, right, bottom, left
    const readout = cfaString(XTRANS_PATTERN, 6, 5, 3);
    const cfa = findPatternShift(readout, 6, crops);
    const data = new Uint16Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) data[y * w + x] = 'RGB'.indexOf(readout[(y % 6) * 6 + (x % 6)]);
    const out = cropAndPad(data, w, visibleRect(w, h, crops), cfa.dy, cfa.dx, 6);
    for (let y = 0; y < out.height; y++) {
      for (let x = 0; x < out.width; x++) expect(out.data[y * out.width + x]).toBe(XTRANS_PATTERN[y % 6][x % 6]);
    }
  });
});

describe('prepareCfa', () => {
  it('equals crop → normalise → pad done the old way, value for value', () => {
    const w = 40, h = 30, crops = new Uint16Array([1, 3, 2, 4]);
    const readout = cfaString(XTRANS_PATTERN, 6, 1, 4);
    const data = noisy(w * h, 3000, 11);
    for (let i = 0; i < data.length; i += 7) data[i] = 16000;
    const raw = {
      data, width: w, height: h, crops, cfaStr: readout, cfaWidth: 6,
      blackLevels: new Uint16Array([500, 510, 520, 0]), whiteLevels: new Uint16Array([16383, 16383, 16383, 16383]),
    } as unknown as RawImage;
    const prep = prepareCfa(raw);

    // Reference: crop, calibrate, normalise per visible colour (float), then pad same-phase.
    const rect = visibleRect(w, h, crops);
    const cfa = findPatternShift(readout, 6, crops);
    const vis = new Uint16Array(rect.width * rect.height);
    for (let y = 0; y < rect.height; y++) vis.set(data.subarray((y + rect.top) * w + rect.left, (y + rect.top) * w + rect.left + rect.width), y * rect.width);
    const whites = calibrateReference(vis, rect.width, rect.height, raw.whiteLevels, cfa);
    expect(Array.from(prep.whiteLevels)).toEqual(whites);
    const lutVis = visibleColorLut(cfa);
    const norm = new Float32Array(vis.length);
    for (let y = 0; y < rect.height; y++) for (let x = 0; x < rect.width; x++) {
      const c = lutVis[(y % 6) * 6 + (x % 6)];
      norm[y * rect.width + x] = (vis[y * rect.width + x] - raw.blackLevels[c]) / (whites[c] - raw.blackLevels[c]);
    }
    const phase = (i: number, pad: number, size: number) => Math.min(i < pad ? (((i - pad) % 6) + 6) % 6 : i - pad, size - 1);
    const pattern = new Uint32Array(XTRANS_PATTERN.flat());
    const got = normalizeRows(prep.data, prep.width, 0, prep.height, pattern, 6, prep.lut);
    expect([prep.width, prep.height, prep.padTop, prep.padLeft]).toEqual([rect.width + cfa.dx, rect.height + cfa.dy, cfa.dy, cfa.dx]);
    for (let y = 0; y < prep.height; y++) for (let x = 0; x < prep.width; x++) {
      const want = norm[phase(y, cfa.dy, rect.height) * rect.width + phase(x, cfa.dx, rect.width)];
      expect(got[y * prep.width + x]).toBe(want);
    }
  });
});

describe('generateTiles', () => {
  it('covers a padded grid with overlapping tiles', () => {
    // stride 24: wPad = ceil(92/24)·24 + 32 = 128, hPad = ceil(52/24)·24 + 32 = 104
    const grid = generateTiles(100, 60, 32, 8);
    expect([grid.wPad, grid.hPad]).toEqual([128, 104]);
    expect(grid.tiles.length).toBe(5 * 4);
    expect(grid.tiles[0]).toEqual({ x: 0, y: 0 });
    expect(grid.tiles.at(-1)).toEqual({ x: 96, y: 72 });
  });
});

describe('makeChannelMasks', () => {
  it('assigns every pixel to exactly one channel with the reference proportions', () => {
    const masks = makeChannelMasks(12, XTRANS_PATTERN, 6);
    const sum = (a: Float32Array) => a.reduce((s, v) => s + v, 0);
    expect(sum(masks.r) + sum(masks.g) + sum(masks.b)).toBe(144);
    expect([sum(masks.r), sum(masks.g), sum(masks.b)]).toEqual([32, 80, 32]);
    const bayer = makeChannelMasks(4, BAYER_PATTERN, 2);
    expect([sum(bayer.r), sum(bayer.g), sum(bayer.b)]).toEqual([4, 8, 4]);
  });
});
