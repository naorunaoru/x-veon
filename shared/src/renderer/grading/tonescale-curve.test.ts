import { describe, it, expect } from 'vitest';
import { evalTonescale, sampleToneCurve } from './tonescale-curve';
import { configFromPreset, configWithOverrides, computeTonescaleParams } from '@/renderer/grading/opendrt-params';

const base = configFromPreset('default');
const cfg = configWithOverrides(base, {}, {});
const ts = computeTonescaleParams(cfg);

describe('evalTonescale', () => {
  it('maps black to ~0 and is bounded in [0,1]', () => {
    expect(evalTonescale(0, cfg, ts)).toBeCloseTo(0, 5);
    for (const x of [0.001, 0.18, 1, 4, 16]) {
      const y = evalTonescale(x, cfg, ts);
      expect(y).toBeGreaterThanOrEqual(0);
      expect(y).toBeLessThanOrEqual(1);
    }
  });
  it('is monotonically increasing in scene input', () => {
    let prev = -1;
    for (let i = 0; i <= 40; i++) {
      const x = 0.18 * Math.pow(2, -8 + (i / 40) * 14);
      const y = evalTonescale(x, cfg, ts);
      expect(y).toBeGreaterThanOrEqual(prev - 1e-6);
      prev = y;
    }
  });
  it('higher contrast steepens the midtone (more separation around middle grey)', () => {
    const lo = configWithOverrides(base, { tn_con: 1.0 }, {});
    const hi = configWithOverrides(base, { tn_con: 1.8 }, {});
    const yLoLow = evalTonescale(0.045, lo, computeTonescaleParams(lo));
    const yLoHigh = evalTonescale(0.72, lo, computeTonescaleParams(lo));
    const yHiLow = evalTonescale(0.045, hi, computeTonescaleParams(hi));
    const yHiHigh = evalTonescale(0.72, hi, computeTonescaleParams(hi));
    expect(yHiHigh - yHiLow).toBeGreaterThan(yLoHigh - yLoLow);
  });
});

describe('sampleToneCurve', () => {
  it('returns n normalized points with x spanning [0,1] and y in [0,1]', () => {
    const pts = sampleToneCurve(cfg, ts, 32);
    expect(pts).toHaveLength(32);
    expect(pts[0].x).toBeCloseTo(0, 6);
    expect(pts[31].x).toBeCloseTo(1, 6);
    for (const p of pts) {
      expect(p.y).toBeGreaterThanOrEqual(0);
      expect(p.y).toBeLessThanOrEqual(1);
    }
  });
});
