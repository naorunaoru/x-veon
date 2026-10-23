import { describe, expect, it } from 'vitest';
import { DEFAULT_LENS_EDIT, effectiveLensEdit, isLensEditModified, lensEditEquals, lensSourceKey } from './edit';

describe('lens edit', () => {
  it('fills defaults for a missing block', () => {
    expect(effectiveLensEdit({})).toEqual(DEFAULT_LENS_EDIT);
    expect(effectiveLensEdit({})).not.toBe(DEFAULT_LENS_EDIT);
  });
  it('is unmodified when absent or equal to the defaults', () => {
    expect(isLensEditModified({})).toBe(false);
    expect(isLensEditModified({ lens: { ...DEFAULT_LENS_EDIT, manual: { ...DEFAULT_LENS_EDIT.manual } } })).toBe(false);
  });
  it.each([
    { source: 'manual' as const }, { distortion: false }, { manual: { ...DEFAULT_LENS_EDIT.manual, midpoint: 0.6 } },
    { source: { lensfun: { file: 'mil-fujifilm.json', model: 'XF35mmF2 R WR' } } },
  ])('is modified by %o', (patch) => {
    expect(isLensEditModified({ lens: { ...DEFAULT_LENS_EDIT, ...patch } })).toBe(true);
  });
  it('compares effective values', () => {
    expect(lensEditEquals({}, { lens: { ...DEFAULT_LENS_EDIT } })).toBe(true);
    expect(lensEditEquals({}, { lens: { ...DEFAULT_LENS_EDIT, ca: false } })).toBe(false);
  });
  it('keys sources', () => {
    expect(lensSourceKey('auto')).toBe('auto');
    expect(lensSourceKey({ lensfun: { file: 'misc.json', model: '23mmF1.4XM' } })).toBe('lensfun:misc.json#23mmF1.4XM');
  });
});
