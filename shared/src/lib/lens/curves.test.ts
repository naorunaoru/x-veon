import { describe, it, expect } from 'vitest';
import type { LfDistortion, LfTCA } from '../lensfun-types';
import { NO_CURVES, evalDistortion, evalTca, evalVignettingGain, huginUnits, lensfunCurves, manualCurves } from './curves';

describe('lensfun models', () => {
  it('ptlens leaves r = 1 fixed and compresses the corner', () => {
    const e = { model: 'ptlens', focal: 35, a: 0, b: -0.02, c: 0 };
    expect(evalDistortion(e, 1)).toBeCloseTo(1, 9);
    expect(evalDistortion(e, 1.8)).toBeCloseTo(1.71936, 5);
  });
  it('poly3 and poly5', () => {
    expect(evalDistortion({ model: 'poly3', focal: 35, k1: 0.01 }, 2)).toBeCloseTo(2.06, 9);
    expect(evalDistortion({ model: 'poly5', focal: 35, k1: 0.01, k2: 0.001 }, 2)).toBeCloseTo(2.112, 9);
  });
  it('TCA linear and poly3 act on the distorted radius', () => {
    expect(evalTca({ model: 'linear', focal: 35, kr: 1.001, kb: 0.999 }, 2)).toEqual({ red: expect.closeTo(2.002, 9), blue: expect.closeTo(1.998, 9) });
    expect(evalTca({ model: 'poly3', focal: 35, br: 0.001, vr: 1, vb: 1 }, 1).red).toBeCloseTo(1.001, 9);
  });
  it('a negative k1 brightens', () => {
    expect(evalVignettingGain({ model: 'pa', focal: 35, aperture: 2, distance: 1000, k1: -0.4, k2: 0.1, k3: 0.02 }, 1)).toBeCloseTo(1 / 0.72, 9);
  });
  it('hugin units: Fuji X on Fuji X, and a full-frame calibration on APS-C', () => {
    expect(huginUnits({ cropfactor: 1.5, calibCropfactor: 1.5, calibAspectRatio: 1.5 })).toEqual({ geometry: expect.closeTo(1.802776, 5), vignetting: 1 });
    expect(huginUnits({ cropfactor: 1.5, calibCropfactor: 1, calibAspectRatio: 1.5 })).toEqual({ geometry: expect.closeTo(1.201850, 5), vignetting: expect.closeTo(2 / 3, 9) });
  });
});
describe('lensfun models, edge cases', () => {
  it('an unknown model changes nothing', () => {
    expect(evalDistortion({ model: 'acm', focal: 35 }, 1.3)).toBe(1.3);
    expect(evalTca({ model: 'acm', focal: 35 }, 1.3)).toEqual({ red: 1.3, blue: 1.3 });
    expect(evalVignettingGain({ model: 'acm', focal: 35, aperture: 2, distance: 1000, k1: -0.4, k2: 0, k3: 0 }, 1)).toBe(1);
  });
  it('missing coefficients take their defaults', () => {
    expect(evalDistortion({ model: 'poly3', focal: 35 }, 1.7)).toBeCloseTo(1.7, 12);
    expect(evalDistortion({ model: 'poly5', focal: 35 }, 1.7)).toBeCloseTo(1.7, 12);
    expect(evalDistortion({ model: 'ptlens', focal: 35 }, 1.7)).toBeCloseTo(1.7, 12);
    expect(evalTca({ model: 'linear', focal: 35 }, 1.7)).toEqual({ red: 1.7, blue: 1.7 });
    expect(evalTca({ model: 'poly3', focal: 35 }, 1.7)).toEqual({ red: 1.7, blue: 1.7 });
  });
  it('TCA poly3 reads the blue coefficients for blue', () => {
    expect(evalTca({ model: 'poly3', focal: 35, bb: 0.002, cb: 0.003, vb: 0.99 }, 2).blue).toBeCloseTo(2 * (0.002 * 4 + 0.003 * 2 + 0.99), 12);
  });
});
describe('lensfunCurves', () => {
  const base = { lensModel: 'L', mount: 'Fujifilm X', cropfactor: 1.5, calibCropfactor: 1.5, calibAspectRatio: 1.5, tca: [], vignetting: [] };
  it('blends the two bracketing focal lengths and clamps outside', () => {
    const p = { ...base, distortion: [{ model: 'poly3', focal: 18, k1: 0.02 }, { model: 'poly3', focal: 55, k1: -0.02 }] };
    const at = (f: number) => lensfunCurves(p, f, 4).distortion!(1);
    expect(at(36.5)).toBeCloseTo((at(18) + at(55)) / 2, 9);
    expect(at(10)).toBeCloseTo(at(18), 9); expect(at(100)).toBeCloseTo(at(55), 9);
    expect(at(18)).toBeCloseTo(evalDistortion(p.distortion[0], 1.802776) / 1.802776, 5);
  });
  it('vignetting: nearest focal, apertures blended in log2 f-number, distance nearest 1000 m', () => {
    const v = (aperture: number, distance: number, k1: number) => ({ model: 'pa', focal: 35, aperture, distance, k1, k2: 0, k3: 0 });
    const p = { ...base, distortion: [], vignetting: [v(2, 1000, -0.5), v(4, 1000, -0.2), v(2, 10, -0.9)] };
    const gain = lensfunCurves(p, 35, 2.8).vignetting!(1);
    expect(gain).toBeGreaterThan(1.25); expect(gain).toBeLessThan(2);
    expect(lensfunCurves(p, 35, 2).vignetting!(1)).toBeCloseTo(2, 9);  // not the 10 m entry (gain 10)
  });
  it('returns null curves for corrections without entries, and for focal 0 with several entries', () => {
    expect(lensfunCurves({ ...base, distortion: [] }, 35, 4).distortion).toBeNull();
    const two = { ...base, distortion: [{ model: 'poly3', focal: 18, k1: 0.02 }, { model: 'poly3', focal: 55, k1: -0.02 }] };
    expect(lensfunCurves(two, 0, 4).distortion).toBeNull();
    expect(lensfunCurves({ ...base, distortion: [two.distortion[0]] }, 0, 4).distortion).not.toBeNull();
  });
  it('takes the entry itself at a calibrated focal length', () => {
    const entry = (focal: number, k1: number) => ({ model: 'poly3', focal, k1 });
    const three = { ...base, distortion: [entry(18, 0.02), entry(35, 0.05), entry(55, -0.02)] };
    expect(lensfunCurves(three, 35, 4).distortion!(1)).toBeCloseTo(lensfunCurves({ ...base, distortion: [entry(35, 0.05)] }, 35, 4).distortion!(1), 12);
    expect(lensfunCurves({ ...base, distortion: [entry(35, 0.05)] }, 100, 4).distortion!(1)).toBeCloseTo(lensfunCurves(three, 35, 4).distortion!(1), 12);
  });
  it('at the centre the ratio is the model\'s constant term, blended like the rest', () => {
    const at0 = (distortion: LfDistortion[], f = 35) => lensfunCurves({ ...base, distortion }, f, 4).distortion!(0);
    expect(at0([{ model: 'poly3', focal: 35, k1: 0.02 }])).toBeCloseTo(0.98, 12);
    expect(at0([{ model: 'ptlens', focal: 35, a: 0, b: -0.02, c: 0 }])).toBeCloseTo(1.02, 12);
    expect(at0([{ model: 'poly5', focal: 35, k1: 0.02, k2: 0.01 }])).toBeCloseTo(1, 12);
    expect(at0([{ model: 'poly3', focal: 18, k1: 0.02 }, { model: 'poly3', focal: 35, k1: 0.05 }], 28)).toBeCloseTo((7 / 17) * 0.98 + (10 / 17) * 0.95, 12);
  });
  it('blends curves of entries with different models, whatever order they come in', () => {
    const a = { model: 'poly3', focal: 18, k1: 0.02 };
    const b = { model: 'ptlens', focal: 55, a: 0, b: -0.03, c: 0 };
    const at = (list: LfDistortion[], f: number) => lensfunCurves({ ...base, distortion: list }, f, 4).distortion!(1);
    const expected = 0.75 * at([a], 18) + 0.25 * at([b], 55);   // 27.25 mm is a quarter of the way from 18 to 55 -> 0.75 / 0.25
    expect(lensfunCurves({ ...base, distortion: [b, a] }, 18 + 0.25 * 37, 4).distortion!(1)).toBeCloseTo(expected, 12);
  });
  it('entries sharing a focal length stay finite', () => {
    const p = { ...base, distortion: [{ model: 'poly3', focal: 35, k1: 0.02 }, { model: 'poly3', focal: 35, k1: 0.04 }, { model: 'poly3', focal: 50, k1: 0 }] };
    for (const f of [20, 35, 42, 50, 80]) expect(Number.isFinite(lensfunCurves(p, f, 4).distortion!(1))).toBe(true);
  });
  it('TCA: brackets by focal length and works in table units', () => {
    const g = Math.hypot(1.5, 1);
    const lin = (focal: number, kr: number, kb: number) => ({ model: 'linear', focal, kr, kb });
    const p = { ...base, distortion: [], tca: [lin(18, 1.002, 0.998), lin(55, 1.000, 1.000)] };
    const c = lensfunCurves(p, 36.5, 4);
    expect(c.caRed!(0.8)).toBeCloseTo(1.001, 9);
    expect(c.caBlue!(0.8)).toBeCloseTo(0.999, 9);
    expect(c.caRed!(0)).toBeCloseTo(1.001, 12);   // the model's constant term, not 1
    const poly = lensfunCurves({ ...base, distortion: [], tca: [{ model: 'poly3', focal: 35, br: 0.001, vr: 1, vb: 1 }] }, 35, 4);
    expect(poly.caRed!(1)).toBeCloseTo(1 + 0.001 * g * g, 12);   // r = g at the corner; r_d / r = b r^2 + v
    expect(poly.caBlue!(1)).toBeCloseTo(1, 12);
    expect(poly.caRed!(0)).toBe(1);   // v_R
    const centre = (tca: LfTCA[]) => lensfunCurves({ ...base, distortion: [], tca }, 35, 4);
    expect(centre([lin(35, 1.001, 0.999)]).caRed!(0)).toBeCloseTo(1.001, 12);
    expect(centre([lin(35, 1.001, 0.999)]).caBlue!(0)).toBeCloseTo(0.999, 12);
    expect(centre([{ model: 'poly3', focal: 35, br: 0.001, cr: 0.002, vr: 1.003, vb: 0.997 }]).caRed!(0)).toBeCloseTo(1.003, 12);
    expect(centre([{ model: 'poly3', focal: 35, vr: 1.003, vb: 0.997 }]).caBlue!(0)).toBeCloseTo(0.997, 12);
  });
  it('vignetting: the radius unit follows the calibration sensor, the focal length is the nearest calibrated one', () => {
    const v = (focal: number, k1: number) => ({ model: 'pa', focal, aperture: 2, distance: 1000, k1, k2: 0, k3: 0 });
    const p = { ...base, calibCropfactor: 1, distortion: [], vignetting: [v(24, -0.5), v(70, -0.1)] };
    expect(lensfunCurves(p, 30, 2).vignetting!(1)).toBeCloseTo(1 / (1 - 0.5 * (2 / 3) ** 2), 12);
    expect(lensfunCurves(p, 60, 2).vignetting!(1)).toBeCloseTo(1 / (1 - 0.1 * (2 / 3) ** 2), 12);
  });
  it('vignetting: an unknown focal length needs the entries to share one calibrated focal length', () => {
    const v = (focal: number, k1: number) => ({ model: 'pa', focal, aperture: 2, distance: 1000, k1, k2: 0, k3: 0 });
    const zoom = { ...base, distortion: [], vignetting: [v(24, -0.5), v(70, -0.1)] };
    const prime = { ...base, distortion: [], vignetting: [v(35, -0.5), { ...v(35, -0.2), aperture: 4 }] };
    for (const f of [0, -1, NaN, Infinity]) {
      expect(lensfunCurves(zoom, f, 2).vignetting).toBeNull();
      expect(lensfunCurves(prime, f, 2).vignetting!(1)).toBeCloseTo(2, 9);
    }
  });
  it('vignetting: an f-number outside the calibrated apertures takes the nearest one', () => {
    const v = (aperture: number, k1: number) => ({ model: 'pa', focal: 35, aperture, distance: 1000, k1, k2: 0, k3: 0 });
    const p = { ...base, distortion: [], vignetting: [v(4, -0.2), v(2, -0.5)] };
    expect(lensfunCurves(p, 35, 11).vignetting!(1)).toBeCloseTo(1.25, 9);
    expect(lensfunCurves(p, 35, 1.4).vignetting!(1)).toBeCloseTo(2, 9);
    expect(lensfunCurves(p, 35, Math.SQRT2 * 2).vignetting!(1)).toBeCloseTo(0.5 * (2 + 1.25), 9);   // f/2.83 is halfway in log2
  });
  it('vignetting: a single entry serves every f-number', () => {
    const p = { ...base, distortion: [], vignetting: [{ model: 'pa', focal: 35, aperture: 2.8, distance: 1000, k1: -0.5, k2: 0, k3: 0 }] };
    for (const f of [1.4, 2.8, 16]) expect(lensfunCurves(p, 35, f).vignetting!(1)).toBeCloseTo(2, 9);
  });
  it('an unknown f-number (0, negative, NaN, infinite) gives no vignetting and leaves the rest alone', () => {
    const entry = { model: 'pa', focal: 35, aperture: 2.8, distance: 1000, k1: -0.5, k2: 0, k3: 0 };
    const one = { ...base, distortion: [{ model: 'poly3', focal: 35, k1: 0.02 }], vignetting: [entry] };
    const two = { ...one, vignetting: [entry, { ...entry, aperture: 5.6 }] };
    for (const f of [0, -2.8, NaN, Infinity]) {
      expect(lensfunCurves(one, 35, f).vignetting).toBeNull();
      expect(lensfunCurves(two, 35, f).vignetting).toBeNull();
      expect(lensfunCurves(one, 35, f).distortion).not.toBeNull();
    }
  });
  it('an unknown focal length (0, negative, NaN, infinite) takes the only entry or none, and never throws', () => {
    const entry = (focal: number) => ({ model: 'poly3', focal, k1: 0.02 });
    const lin = (focal: number) => ({ model: 'linear', focal, kr: 1.001, kb: 0.999 });
    const one = { ...base, distortion: [entry(35)], tca: [lin(35)] };
    const two = { ...base, distortion: [entry(18), entry(55)], tca: [lin(18), lin(55)] };
    for (const f of [0, -35, NaN, Infinity, -Infinity]) {
      const single = lensfunCurves(one, f, 4);
      expect(single.distortion!(1)).toBeCloseTo(lensfunCurves(one, 35, 4).distortion!(1), 12);
      expect(single.caRed!(1)).toBeCloseTo(1.001, 12);
      const several = lensfunCurves(two, f, 4);
      expect(several.distortion).toBeNull(); expect(several.caRed).toBeNull(); expect(several.caBlue).toBeNull();
    }
  });
});
describe('manualCurves', () => {
  it('maps the sliders to the spec formulas', () => {
    const c = manualCurves({ distortion: 10, caRed: 5, caBlue: -5, vignetting: 1, midpoint: 0.5 }, 1000);
    expect(c.distortion!(1)).toBeCloseTo(0.9, 9);
    expect(c.caRed!(1)).toBeCloseTo(1.005, 9); expect(c.caBlue!(1)).toBeCloseTo(0.995, 9);
    expect(c.vignetting!(1)).toBeCloseTo(2, 9); expect(c.vignetting!(0.5)).toBeCloseTo(Math.SQRT2, 9);
  });
  it('only the sliders that moved give curves', () => {
    const c = manualCurves({ distortion: 0, caRed: 0, caBlue: -2, vignetting: 0, midpoint: 0.5 }, 500);
    expect(c.distortion).toBeNull(); expect(c.caRed).toBeNull(); expect(c.vignetting).toBeNull();
    expect(c.caBlue!(0.3)).toBeCloseTo(1 - 2 / 500, 12);
  });
  it('negative distortion pushes the source out, and the midpoint sits at half the corner EV', () => {
    const c = manualCurves({ distortion: -20, caRed: 0, caBlue: 0, vignetting: 2, midpoint: 0.3 }, 1000);
    expect(c.distortion!(0)).toBe(1);
    expect(c.distortion!(1)).toBeCloseTo(1.2, 12);
    expect(c.vignetting!(0)).toBe(1);
    expect(c.vignetting!(1)).toBeCloseTo(4, 9);
    expect(c.vignetting!(0.3)).toBeCloseTo(2, 9);
  });
  it('is all null at the defaults', () => {
    expect(manualCurves({ distortion: 0, caRed: 0, caBlue: 0, vignetting: 0, midpoint: 0.5 }, 1000)).toEqual(NO_CURVES);
  });
});
