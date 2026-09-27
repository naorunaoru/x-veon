import { describe, expect, it } from 'vitest';
import { configFromPreset, configWithOverrides, computeTonescaleParams, OPENDRT_LIMITS } from './opendrt-params';
import { EXPERT_GROUPS, RANGE_KEY } from '@/components/hud/panels/rendering/constants';

describe('OpenDRT parameter domains', () => {
  it('normalizes old singular overrides without mutating saved values', () => {
    const overrides = { rs_sa: 1, pt_rng_low: 0, pt_rng_high: 0, ptm_low_st: 0, ptm_high_st: 0, hs_rgb_rng: 0 };
    const cfg = configWithOverrides(configFromPreset('default'), overrides);
    for (const key of Object.keys(overrides) as (keyof typeof overrides)[]) {
      expect(cfg[key]).toBe(key === 'rs_sa' ? 0.6 : OPENDRT_LIMITS[key][0]);
    }
    expect(overrides.rs_sa).toBe(1);
    expect(overrides.pt_rng_low).toBe(0);
  });
  it('keeps tone constants finite for malformed saved tone values', () => {
    const cfg = configWithOverrides(configFromPreset('default'), {
      tn_lg: NaN, tn_con: 0, tn_off: -0.18, tn_toe: -1, peak_luminance: Infinity,
    });
    expect(Object.values(computeTonescaleParams(cfg)).every(Number.isFinite)).toBe(true);
  });
  it('keeps every range control inside the validated domain', () => {
    for (const row of EXPERT_GROUPS.flatMap(g => g.rows)) {
      if (row.kind !== 'slider' || !(row.key in OPENDRT_LIMITS)) continue;
      const limits = OPENDRT_LIMITS[row.key as keyof typeof OPENDRT_LIMITS];
      expect(row.min).toBeGreaterThanOrEqual(limits[0]);
      expect(row.max).toBeLessThanOrEqual(limits[1]);
    }
    expect(RANGE_KEY.hue.min).toBe(0.25);
    expect(RANGE_KEY.pur.min).toBe(0.1);
  });
});
