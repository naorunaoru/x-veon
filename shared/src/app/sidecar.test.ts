import { expect, it } from 'vitest';
import { decodeSidecar, encodeSidecar, isKnownProperty } from './sidecar';
import { defaultPhotoEdit } from './photo-edit';
import type { PhotoEdit } from '@/host';
import { DEFAULT_LENS_EDIT } from '@/lib/lens/edit';
import { configFromPreset } from '@/renderer/grading/opendrt-params';

const DRT = configFromPreset('default');

it('encodes nothing for a default edit and rating 0', () => {
  expect(encodeSidecar({ edit: defaultPhotoEdit(), rating: 0 })).toEqual({});
});

it('encodes a rating alone without a schema version', () => {
  expect(encodeSidecar({ edit: defaultPhotoEdit(), rating: 3 })).toEqual({ 'xmp:Rating': '3' });
});

it('round-trips a full edit with a neural model, a boolean override and a lens block', () => {
  expect(typeof DRT.cwp).toBe('number');
  expect(typeof DRT.ptl_enable).toBe('boolean');
  const edit: PhotoEdit = {
    ...defaultPhotoEdit(),
    lookPreset: 'umbra',
    demosaicMethod: 'neural-net',
    model: { size: 'M', sha256: 'ab'.repeat(32) },
    preProcessOverrides: { exposure: 0.5 },
    openDrtOverrides: { cwp: 0.4, ptl_enable: !DRT.ptl_enable },
    lens: { ...DEFAULT_LENS_EDIT, source: { lensfun: { file: 'f.xml', model: 'Lens' } }, ca: false, manual: { ...DEFAULT_LENS_EDIT.manual, midpoint: 0.7 } },
  };
  const encoded = encodeSidecar({ edit, rating: -1 });
  expect(encoded).toMatchObject({
    'xveon:SchemaVersion': '1', 'xveon:Look': 'umbra', 'xveon:ModelSize': 'M', 'xveon:LensSource': 'lensfun',
    'xveon:LensCa': 'False', 'xveon:LensManualMidpoint': '0.7', 'xmp:Rating': '-1',
  });
  expect(decodeSidecar(encoded)).toEqual({ edit, rating: -1, schemaVersion: 1, unknown: {} });
});

it('keeps unknown xveon properties through encode and decode', () => {
  expect(decodeSidecar({ 'xveon:Future': 'x', 'xveon:Look': 'umbra' }).unknown).toEqual({ 'xveon:Future': 'x' });
  expect(encodeSidecar({ edit: defaultPhotoEdit(), rating: 0 }, { 'xveon:Future': 'x' })).toEqual({ 'xveon:SchemaVersion': '1', 'xveon:Future': 'x' });
  expect(encodeSidecar({ edit: defaultPhotoEdit(), rating: 0 }, { 'dc:creator': 'me' })).toEqual({ 'dc:creator': 'me' });
});

it('rejects invalid values with the property name', () => {
  const cases: [Record<string, string>, RegExp][] = [
    [{ 'xveon:Look': 'nope' }, /^Look: invalid preset/],
    [{ 'xveon:Exposure': 'abc' }, /^Exposure: invalid number/],
    [{ 'xveon:Odrt_ptl_enable': 'yes' }, /invalid boolean/],
    [{ 'xveon:ModelSize': 'M' }, /^ModelHash: missing hash/],
    [{ 'xveon:SchemaVersion': '0' }, /^SchemaVersion: invalid version/],
    [{ 'xmp:Rating': '7' }, /^xmp:Rating: invalid rating/],
    [{ 'xmp:Rating': '2.5' }, /^xmp:Rating: invalid rating/],
    [{ 'xmp:Rating': 'x' }, /^xmp:Rating: invalid rating/],
  ];
  for (const [properties, message] of cases) expect(() => decodeSidecar(properties)).toThrow(message);
});

it('reads a newer schema version through', () => {
  expect(decodeSidecar({ 'xveon:SchemaVersion': '2' }).schemaVersion).toBe(2);
});

it('knows the property set', () => {
  for (const name of ['xveon:SchemaVersion', 'xveon:Odrt_cwp', 'xveon:LensManualMidpoint', 'xmp:Rating']) expect(isKnownProperty(name)).toBe(true);
  for (const name of ['xveon:Odrt_bogus', 'xveon:Future', 'xmp:Label', 'dc:creator']) expect(isKnownProperty(name)).toBe(false);
});

it('formats numbers with String(n) and rejects non-finite ones', () => {
  const edit = (exposure: number): PhotoEdit => ({ ...defaultPhotoEdit(), preProcessOverrides: { exposure } });
  expect(encodeSidecar({ edit: edit(0.1 + 0.2), rating: 0 })['xveon:Exposure']).toBe('0.30000000000000004');
  expect(() => encodeSidecar({ edit: edit(Infinity), rating: 0 })).toThrow(/Exposure: non-finite number/);
});
