import { describe, expect, it } from 'vitest';
import { linearSpline } from './curves';
import { fujiCurves, fujiFrameCentre, fujiHas, parseFujiTables, type FujiKnots } from './fuji-tables';
import { K9, K11, XE1, XT3 } from './fuji-fixtures';

describe('parseFujiTables', () => {
  it('parses the new layout', () => {
    const k = parseFujiTables(XT3)!;
    expect(k.knots).toHaveLength(9); expect(k.knots[0]).toBeCloseTo(0.35352, 5);
    expect(k.distortion[8]).toBeCloseTo(-8.895065308, 6);
    expect(k.caRed[0]).toBeCloseTo(-0.001251220703, 9); expect(k.caBlue[8]).toBeCloseTo(0.001525878906, 9);
    expect(k.halfDiagPx).toBeCloseTo(3749, 0);
    expect(fujiHas(k)).toEqual({ distortion: true, ca: true, vignetting: false });
  });
  it('parses the old layout with a synthetic zero CA knot', () => {
    const k = parseFujiTables(XE1)!;
    expect(k.knots).toEqual(K11);
    expect(k.caRed[0]).toBe(0); expect(k.caRed[1]).toBeCloseTo(0.000209, 9); expect(k.caBlue[10]).toBeCloseTo(0.001068, 9);
    expect(k.vignetting[10]).toBeCloseTo(47.05, 6);
    expect(k.halfDiagPx).toBeCloseTo(2942, 0);
    expect(fujiHas(k)).toEqual({ distortion: true, ca: true, vignetting: true });
  });
  it('rejects knot mismatches and unknown counts', () => {
    const ca = [...XT3.ca]; ca[1] = 0.4;
    expect(parseFujiTables({ ...XT3, ca })).toBeNull();
    expect(parseFujiTables({ ...XT3, vignetting: XT3.vignetting.slice(0, 18) })).toBeNull();
    expect(parseFujiTables({ distortion: [], ca: [], vignetting: [] })).toBeNull();
  });
  it('rejects tags of different layouts and old-layout CA knots that disagree', () => {
    expect(parseFujiTables({ ...XT3, ca: XE1.ca })).toBeNull();
    const ca = [...XE1.ca]; ca[10] = 0.95;
    expect(parseFujiTables({ ...XE1, ca })).toBeNull();
  });
  it('reports a zero half-diagonal when the first tag value is 0', () => {
    expect(parseFujiTables({ ...XT3, distortion: [0, ...XT3.distortion.slice(1)] })!.halfDiagPx).toBe(0);
  });
});

describe('fujiHas and fujiCurves', () => {
  it('treats all-zero distortion and all-100 vignetting as absent', () => {
    const k = parseFujiTables({ ...XT3, distortion: [416.5555556, ...K9, 0, 0, 0, 0, 0, 0, 0, 0, 0] })!;
    expect(fujiHas(k).distortion).toBe(false);
    expect(fujiCurves(k).distortion).toBeNull();
  });
  it('keeps the vignetting curve when only the distortion is trivial', () => {
    const k = parseFujiTables({ ...XT3, distortion: [416.5555556, ...K9, 0, 0, 0, 0, 0, 0, 0, 0, 0], vignetting: [416.5555556, ...K9, 98, 96, 94, 92, 90, 88, 86, 84, 82] })!;
    const c = fujiCurves(k);
    expect(c.distortion).toBeNull();
    expect(c.vignetting!(K9[8])).toBeCloseTo(100 / 82, 9);
    expect(c.vignetting!(0)).toBe(1);
    expect(c.caRed).not.toBeNull();
  });
  it('treats all-zero CA as absent', () => {
    const k = parseFujiTables({ ...XT3, ca: [416.5555556, ...K9, ...new Array(18).fill(0), 416.5555556] })!;
    expect(fujiHas(k).ca).toBe(false);
    const c = fujiCurves(k);
    expect(c.caRed).toBeNull();
    expect(c.caBlue).toBeNull();
  });
  it('honours a caller-supplied selection of corrections', () => {
    const k = parseFujiTables(XE1)!;
    const c = fujiCurves(k, { distortion: false, ca: true, vignetting: false });
    expect([c.distortion, c.vignetting]).toEqual([null, null]);
    expect(c.caRed).not.toBeNull();
  });
  it('builds the old layout curves through their own zero knot', () => {
    const c = fujiCurves(parseFujiTables(XE1)!);
    const m = 1 - 0.06542;
    expect(c.distortion!(0)).toBe(1);
    expect(c.distortion!(1 / m)).toBeCloseTo(m, 9);         // the source corner
    expect(c.caRed!(0)).toBe(1);
    expect(c.caRed!(0.1)).toBeCloseTo(1.000209, 9);
    expect(c.caBlue!(1)).toBeCloseTo(1.001068, 9);
    expect(c.vignetting!(1)).toBeCloseTo(100 / 47.05, 9);
  });
  it('re-indexes distortion from source to output radius and clamps beyond the last knot', () => {
    const k: FujiKnots = { knots: [0.5, 1], distortion: [-10, -10], caRed: [0.001, 0.001], caBlue: [-0.002, -0.002], vignetting: [50, 50], halfDiagPx: 1000 };
    const c = fujiCurves(k);
    expect(c.distortion!(1 / 0.9)).toBeCloseTo(0.9, 6);   // the source corner (r_s = 1, m = 0.9) sits at output radius 1.111
    expect(c.distortion!(0)).toBeCloseTo(1, 6);            // synthetic zero knot: no correction at the centre
    expect(c.distortion!(5)).toBeCloseTo(0.9, 6);          // clamp
    expect(c.caRed!(0.7)).toBeCloseTo(1.001, 9); expect(c.caBlue!(0.7)).toBeCloseTo(0.998, 9);
    expect(c.vignetting!(0.75)).toBeCloseTo(2, 6);         // 100 / 50
  });
});

describe('fujiFrameCentre', () => {
  it('resolves the camera frame centre only when the full size matches the decode', () => {
    const frame = [4182, 6384, 13, 6, 4160, 6240];
    expect(fujiFrameCentre(frame, 6384, 4182, { left: 0, top: 0, width: 6384, height: 4182 })).toEqual([6 + 3120, 13 + 2080]);
    expect(fujiFrameCentre(frame, 6384, 4182, { left: 10, top: 4, width: 6300, height: 4170 })).toEqual([3116, 2089]);
    expect(fujiFrameCentre(frame, 6276, 4140, { left: 0, top: 0, width: 6276, height: 4140 })).toBeNull();
    expect(fujiFrameCentre([], 6384, 4182, { left: 0, top: 0, width: 6384, height: 4182 })).toBeNull();
  });
  it('rejects a frame without a cropped size', () => {
    const visible = { left: 0, top: 0, width: 6, height: 4 };
    expect(fujiFrameCentre([0, 0, 0, 0, 0, 0], 6, 4, visible)).toBeNull();
    expect(fujiFrameCentre([4, 6, 1, 1, 0, 4], 6, 4, visible)).toBeNull();
    expect(fujiFrameCentre([4, 6, 1, 1, 2], 6, 4, visible)).toBeNull();
  });
});

describe('linearSpline', () => {
  it('interpolates between knots, hits them exactly and holds the ends outside', () => {
    const f = linearSpline([0, 1, 3], [10, 20, 40]);
    expect(f(0.5)).toBe(15);
    expect(f(1)).toBe(20);
    expect(f(2)).toBe(30);
    expect(f(-1)).toBe(10);
    expect(f(7)).toBe(40);
  });
  it('is constant through a single knot', () => {
    const f = linearSpline([0.5], [3]);
    expect([f(0), f(0.5), f(2)]).toEqual([3, 3, 3]);
  });
});
