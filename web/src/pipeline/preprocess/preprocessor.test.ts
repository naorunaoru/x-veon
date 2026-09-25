import { describe, it, expect } from 'vitest';
import {
  cropToVisible, findPatternShift, channelClips, calibrateWhiteLevels,
  normalizeRawCfa, padToAlignment, generateTiles, makeChannelMasks, visibleColorLut,
} from './preprocessor';
import { XTRANS_PATTERN, BAYER_PATTERN } from '../constants';

function cfaString(pattern: readonly (readonly number[])[], period: number, dy: number, dx: number): string {
  let s = '';
  for (let y = 0; y < period; y++) {
    for (let x = 0; x < period; x++) s += 'RGB'[pattern[(y + dy) % period][(x + dx) % period]];
  }
  return s;
}

describe('cropToVisible', () => {
  it('returns the input untouched with zero crops', () => {
    const data = new Uint16Array([1, 2, 3, 4]);
    const out = cropToVisible(data, 2, 2, new Uint16Array([0, 0, 0, 0]));
    expect(out.data).toBe(data);
    expect([out.width, out.height]).toEqual([2, 2]);
  });
  it('crops top/right/bottom/left', () => {
    const data = new Uint16Array([...Array(16).keys()]);  // 4×4
    const out = cropToVisible(data, 4, 4, new Uint16Array([1, 1, 1, 1]));
    expect([out.width, out.height]).toEqual([2, 2]);
    expect(Array.from(out.data)).toEqual([5, 6, 9, 10]);
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

describe('calibrateWhiteLevels', () => {
  it('adopts a measured saturation that sits below the metadata white level', () => {
    // 200×200 RGGB: red saturates at 4000 on 10% of its pixels; green and blue each have a single
    // bright outlier (not above the 0.01% threshold), so they keep the metadata level.
    const w = 200, h = 200;
    const data = new Uint16Array(w * h).fill(100);
    data[1] = 500; data[w + 1] = 500;
    for (let y = 0; y < h; y += 2) for (let x = 0; x < w; x += 20) data[y * w + x] = 4000;
    const out = calibrateWhiteLevels(data, w, h, new Uint16Array([16383, 16383, 16383, 16383]), RGGB);
    expect(Array.from(out)).toEqual([4000, 16383, 16383, 16383]);
  });
  it('keeps the metadata level when the measured maximum is at or above it', () => {
    const data = new Uint16Array(16).fill(1000);
    data[0] = 1200;
    expect(Array.from(calibrateWhiteLevels(data, 4, 4, new Uint16Array([1000]), RGGB))).toEqual([1000, 1000, 1000, 1000]);
  });
});

describe('normalizeRawCfa', () => {
  it('normalises each photosite with the black and white level of its colour', () => {
    // RGGB: R, G, G, B. Levels are in RGBE order, as rawloader reports them.
    const data = new Uint16Array([110, 220, 220, 330]);
    const out = normalizeRawCfa(data, 2, 2, new Uint16Array([10, 20, 30, 40]), new Uint16Array([210, 220, 230, 240]), RGGB);
    expect(Array.from(out)).toEqual([0.5, 1.0, 1.0, 1.5]);
  });
  it('ignores the unused E slot (a Canon black level of 0 there must not reach blue)', () => {
    // Canon masked-area blacks: [R, G, B, E=0]. A black frame must normalise to 0 everywhere.
    const blacks = new Uint16Array([1024, 1027, 1030, 0]);
    const whites = new Uint16Array([15600, 15600, 15600, 15600]);
    const out = normalizeRawCfa(new Uint16Array([1024, 1027, 1027, 1030]), 2, 2, blacks, whites, RGGB);
    expect(Array.from(out)).toEqual([0, 0, 0, 0]);
  });
  it('follows the X-Trans colour layout, not 2×2 positions', () => {
    const cfa = findPatternShift(cfaString(XTRANS_PATTERN, 6, 0, 0), 6, new Uint16Array(4));
    const lut = visibleColorLut(cfa);
    const data = new Uint16Array(36);
    for (let i = 0; i < 36; i++) data[i] = [100, 200, 300][lut[i]];
    const out = normalizeRawCfa(data, 6, 6, new Uint16Array([100, 200, 300, 0]), new Uint16Array([1100, 1200, 1300, 0]), cfa);
    expect(Array.from(out).every((v) => v === 0)).toBe(true);
  });
});

describe('padToAlignment', () => {
  it('returns the input for a zero shift', () => {
    const cfa = new Float32Array([1, 2, 3, 4]);
    expect(padToAlignment(cfa, 2, 2, 0, 0).data).toBe(cfa);
  });
  it('mirrors the top and left edges by the shift', () => {
    const cfa = new Float32Array([1, 2, 3, 4, 5, 6]);  // 3×2
    const out = padToAlignment(cfa, 3, 2, 1, 2);
    expect([out.width, out.height, out.padTop, out.padLeft]).toEqual([5, 3, 1, 2]);
    expect(Array.from(out.data)).toEqual([
      2, 1, 1, 2, 3,
      2, 1, 1, 2, 3,
      5, 4, 4, 5, 6,
    ]);
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
