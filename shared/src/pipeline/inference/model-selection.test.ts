import { describe, expect, it } from 'vitest';
import { resolveModel } from './model-selection';
const manifest = {
  xtrans_w16_base: { source_sha256: 'xs', base_width: 16 },
  xtrans_w32_base: { source_sha256: 'xm', base_width: 32 },
  bayer_w16_base: { source_sha256: 'bs', base_width: 16 },
};
describe('per-photo model selection', () => {
  it('matches the exact checkpoint within the decoded CFA regardless of default size or variant preference', () => {
    expect(resolveModel(manifest, 'xtrans', { size: 'S', sha256: 'xs' }, 'M')).toMatchObject({
      key: 'xtrans_w16_base',
      model: { size: 'S', sha256: 'xs' },
      note: null,
    });
  });
  it('falls back to the recorded size, then default size, and explains the mismatch', () => {
    expect(resolveModel(manifest, 'xtrans', { size: 'S', sha256: 'old' }, 'M')).toMatchObject({
      key: 'xtrans_w16_base',
      model: { size: 'S', sha256: 'xs' },
      note: expect.stringContaining('different model'),
    });
    expect(resolveModel(manifest, 'xtrans', { size: 'L', sha256: 'old' }, 'M')).toMatchObject({
      key: 'xtrans_w32_base',
      note: expect.stringContaining('different model'),
    });
  });
  it('uses the default for a missing identity and never resolves a checkpoint from the wrong CFA', () => {
    expect(resolveModel(manifest, 'xtrans', null, 'M').model.sha256).toBe('xm');
    expect(resolveModel(manifest, 'bayer', { size: 'S', sha256: 'xs' }, 'M').model.sha256).toBe('bs');
  });
  it('considers only base entries, by size and by recorded checkpoint', () => {
    const withOther = { ...manifest, xtrans_w16_hl: { source_sha256: 'xh', base_width: 16 } };
    expect(resolveModel(withOther, 'xtrans', null, 'S').key).toBe('xtrans_w16_base');
    expect(resolveModel(withOther, 'xtrans', { size: 'S', sha256: 'xh' }, 'S')).toMatchObject({
      key: 'xtrans_w16_base',
      note: expect.stringContaining('different model'),
    });
    expect(() => resolveModel({ xtrans_w16_hl: { source_sha256: 'xh', base_width: 16 } }, 'xtrans', null, 'S')).toThrow(
      'No S model',
    );
  });
  it('fails clearly if no permitted model exists or the shipped model has no identity', () => {
    expect(() => resolveModel(manifest, 'bayer', null, 'L')).toThrow('No L model');
    expect(() => resolveModel({ xtrans_w16_base: { base_width: 16 } }, 'xtrans', null, 'S')).toThrow(
      'checkpoint hash',
    );
  });
});
