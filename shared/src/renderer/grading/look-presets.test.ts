import { describe, expect, it } from 'vitest';
import { configFromPreset, configWithOverrides, computeTonescaleParams, LOOK_PRESETS, TONESCALE_PRESETS } from './opendrt-params';
import { sampleToneCurve } from './tonescale-curve';
import type { LookPreset } from '@/lib/types';

describe('complete look presets', () => {
  it.each(['low-contrast', 'medium-contrast', 'aces-2', 'marvelous'] as const)('%s has its own tone and a fixed Default colour foundation', (id) => {
    const cfg = configFromPreset(id);
    const baseline = configFromPreset('default');
    expect(cfg).toEqual({ ...baseline, ...TONESCALE_PRESETS[id].overrides });
    expect(configWithOverrides(cfg, {})).toMatchObject(cfg);
  });

  it('offers distinct complete looks and finite tone curves for SDR and HDR', () => {
    const presets = Object.keys(LOOK_PRESETS) as LookPreset[];
    const configurations = presets.map((id) => JSON.stringify(configFromPreset(id)));
    expect(new Set(configurations).size).toBe(presets.length);
    for (const id of presets) {
      for (const headroom of [undefined, 4]) {
        const cfg = configWithOverrides(configFromPreset(id, headroom), {});
        expect(cfg.peak_luminance).toBe(headroom ? 400 : 100);
        const points = sampleToneCurve(cfg, computeTonescaleParams(cfg), 100);
        expect(points.every(({ y }) => Number.isFinite(y) && y >= 0)).toBe(true);
      }
    }
  });

  it('returns independent configurations so preview and export cannot mutate a look', () => {
    const cfg = configFromPreset('marvelous', 4);
    cfg.tn_con = 9;
    expect(configFromPreset('marvelous')).toMatchObject({ tn_con: 1.5, peak_luminance: 100 });
  });
  // Independently transcribed from the official v1.0.0 release, not evolving main.
  it.each([
    ['opendrt-v1-default', { tn_lg: 11.1, hs_r: 0.35, hs_g: 0.25, hs_rgb_rng: 0.6, hc_r: 0.6 }],
    ['opendrt-v1-colorful', { tn_lg: 11.1, tn_con: 1.3, tn_toe: 0.005, rs_rw: 0.15, pt_g: 0.8 }],
    ['opendrt-v1-umbra', { tn_lg: 6, rs_sa: 0.45, rs_rw: 0.1, rs_bw: 0.35, cwp: 1, cwp_rng: 0.8 }],
    ['opendrt-v1-base', { tn_lg: 11.1, tn_off: 0, hs_rgb_enable: false, brl_enable: false }],
  ] as const)('preserves the official %s look parameters', (id, expected) => {
    expect(configFromPreset(id)).toMatchObject(expected);
  });

});
