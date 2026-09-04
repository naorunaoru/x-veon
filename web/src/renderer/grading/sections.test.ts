import { describe, it, expect } from 'vitest';
import { isSectionModified, SECTION_KEYS } from './sections';

describe('isSectionModified', () => {
  it('exposure is modified when exposure pre-override is present', () => {
    expect(isSectionModified('exposure', {}, { exposure: 0.5 })).toBe(true);
  });
  it('exposure is modified when tn_con drt-override is present', () => {
    expect(isSectionModified('exposure', { tn_con: 1.0 }, {})).toBe(true);
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
