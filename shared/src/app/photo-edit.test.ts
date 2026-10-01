import { expect, it } from 'vitest';
import { defaultPhotoEdit, effectiveMethod, isDefaultPhotoEdit } from './photo-edit';

it('keeps an untouched edit empty and recognizes each kind of real edit', () => {
  const empty = defaultPhotoEdit();
  expect(empty).toEqual({ version: 1, lookPreset: 'default', openDrtOverrides: {}, preProcessOverrides: {}, demosaicMethod: null, model: null });
  expect(isDefaultPhotoEdit(empty)).toBe(true);
  expect(isDefaultPhotoEdit({ ...empty, lookPreset: 'umbra' })).toBe(false);
  expect(isDefaultPhotoEdit({ ...empty, preProcessOverrides: { exposure: 1 } })).toBe(false);
  expect(isDefaultPhotoEdit({ ...empty, openDrtOverrides: { cwp: 0.4 } })).toBe(false);
  expect(isDefaultPhotoEdit({ ...empty, demosaicMethod: 'dht' })).toBe(false);
  expect(isDefaultPhotoEdit({ ...empty, model: { size: 'S', sha256: 'used' } })).toBe(false);
});

it('chooses a CFA-compatible processing method without mutating the edit', () => {
  const empty = defaultPhotoEdit();
  expect(effectiveMethod(empty, 'markesteijn3', 'bayer')).toBe('neural-net');
  expect(effectiveMethod(empty, 'markesteijn3', 'xtrans')).toBe('markesteijn3');
  expect(effectiveMethod({ ...empty, demosaicMethod: 'dht' }, 'neural-net', 'xtrans')).toBe('dht');
  expect(empty.demosaicMethod).toBeNull();
});
