import { describe, it, expect } from 'vitest';
import { isSectionModified, SECTION_KEYS, modifiedDrtKeys, RENDERING_KEYS } from './sections';
import { configFromPreset } from './opendrt-params';

describe('isSectionModified', () => {
  it('exposure is modified when exposure pre-override is present', () => {
    expect(isSectionModified('exposure', {}, { exposure: 0.5 })).toBe(true);
  });
  it('contrast belongs to Rendering, not Exposure', () => {
    expect(isSectionModified('exposure', { tn_con: 1.0 }, {})).toBe(false);
    expect(isSectionModified('advanced', { tn_con: 1.0 }, {})).toBe(true);
  });
  it('exposure is not modified with no relevant keys', () => {
    expect(isSectionModified('exposure', { brl_r: 0.2 }, {})).toBe(false);
  });
  it('whiteBalance tracks wb_temp / wb_tint', () => {
    expect(isSectionModified('whiteBalance', {}, { wb_tint: -0.2 })).toBe(true);
    expect(isSectionModified('whiteBalance', {}, { exposure: 1 })).toBe(false);
  });
  it('advanced (Rendering) tracks the OpenDRT keys it owns', () => {
    expect(isSectionModified('advanced', { brl_g: -0.1 }, {})).toBe(true);
  });
  it('sections without a key map (settings) are never modified here', () => {
    expect(isSectionModified('settings', {}, { exposure: 1 })).toBe(false);
  });
  it('exposes the key map for reset', () => {
    expect(SECTION_KEYS.exposure?.pre).toContain('exposure');
    expect(SECTION_KEYS.advanced?.drt).toEqual(expect.arrayContaining(['brl_r', 'brl_g', 'brl_b']));
  });
});


describe('rendering modification tracking', () => {
  it('compares against the selected look, ignoring numeric noise', () => {
    const base = configFromPreset('umbra');
    expect(modifiedDrtKeys(base, { tn_con: base.tn_con + 1e-8 })).toEqual([]);
    expect(modifiedDrtKeys(base, { cwp: 0 })).toContain('cwp');
  });

  it('detects modules enabled by an override even when its numeric value matches the baseline', () => {
    const base = configFromPreset('aces-2');
    expect(modifiedDrtKeys(base, { tn_lcon: base.tn_lcon })).toContain('tn_lcon_enable');
  });

  it('covers every rendering parameter exactly once for section resets', () => {
    const keys = Object.values(RENDERING_KEYS).flat();
    expect([...keys].sort()).toEqual(Object.keys(configFromPreset('default')).sort());
    expect(new Set(keys).size).toBe(keys.length);
  });
});
