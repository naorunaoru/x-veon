import { expect, it } from 'vitest';
import { fakePhoto } from '@/test/fake-host';
import { editPhoto, fromLibraryPhoto, processingKey } from './photo';
import { defaultEdit } from '@/test/fake-host';
const actual = { size: 'S' as const, sha256: 'shipped' };
function processedPhoto() {
  const file = fromLibraryPhoto(fakePhoto());
  file.edit = { ...file.edit, demosaicMethod: 'neural-net', model: { size: 'S', sha256: 'missing' } };
  file.actualModel = actual;
  file.status = 'done';
  file.resultMethod = 'neural-net';
  file.processedKey = processingKey(file, { demosaicMethod: 'neural-net', modelSize: 'S' });
  return file;
}
it('adopts the rendered model on the first grade edit without requesting another inference', () => {
  const file = processedPhoto();
  const edited = editPhoto(file, { preProcessOverrides: { exposure: 1 } }, { demosaicMethod: 'neural-net' });
  expect(edited.edit.model).toEqual(actual);
  expect(edited.processedKey).toBe(processingKey(edited, { demosaicMethod: 'neural-net', modelSize: 'S' }));
});
it('keeps a real method or model change pending for reprocessing', () => {
  const file = processedPhoto();
  for (const patch of [
    { demosaicMethod: 'dht' as const },
    { model: { size: 'M' as const, sha256: 'new' } },
  ]) {
    const edited = editPhoto(file, patch, { demosaicMethod: 'neural-net' });
    expect(edited.processedKey).not.toBe(
      processingKey(edited, { demosaicMethod: 'neural-net', modelSize: 'S' }),
    );
  }
});
it('does not acknowledge a result from an earlier requested model', () => {
  const file = processedPhoto();
  file.processedKey = 'neural-net:S:earlier';
  const edited = editPhoto(file, { preProcessOverrides: { exposure: 1 } }, { demosaicMethod: 'neural-net' });
  expect(edited.processedKey).toBe('neural-net:S:earlier');
});

it('records the result method and actual model when an untouched processed photo is edited', () => {
  const file = processedPhoto();
  file.edit = defaultEdit();
  const edited = editPhoto(file, { preProcessOverrides: { exposure: 1 } }, { demosaicMethod: 'dht' });
  expect(edited.edit.demosaicMethod).toBe('neural-net');
  expect(edited.edit.model).toEqual(actual);
});

it('records the effective method and defers a model when the photo has not processed', () => {
  const file = fromLibraryPhoto(fakePhoto());
  const edited = editPhoto(file, { lookPreset: 'umbra' }, { demosaicMethod: 'neural-net' });
  expect(edited.edit.demosaicMethod).toBe('neural-net');
  expect(edited.edit.model).toBeNull();
  expect(edited.modelNeedsResolution).toBe(true);
});

it('tracks a new default method for an untouched photo', () => {
  const file = fromLibraryPhoto(fakePhoto());
  expect(processingKey(file, { demosaicMethod: 'dht', modelSize: 'S' })).toBe('dht');
  expect(processingKey(file, { demosaicMethod: 'neural-net', modelSize: 'S' })).toBe('neural-net:S:');
});

it('uses a restored neural result identity on the first edit after reload', () => {
  const photo = fakePhoto();
  photo.facts.resultMethod = 'neural-net';
  photo.facts.resultMeta = {
    exportData: { width: 4, height: 2, xyzToCam: null, wbCoeffs: [1, 1, 1], camToXyz: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], orientation: 'Normal' },
    metadata: { make: 'F', model: 'X', width: 4, height: 2, tileCount: 1, inferenceTime: 0, backend: 'webgpu', exposureBias: 0, lensModel: 'L', focalLength: 0, fNumber: 0, colorTemp: 0, tint: 0, modelIdentity: actual },
  };
  const file = fromLibraryPhoto(photo);
  const edited = editPhoto(file, { lookPreset: 'umbra' }, { demosaicMethod: 'dht' });
  expect(edited.edit.demosaicMethod).toBe('neural-net');
  expect(edited.edit.model).toEqual(actual);
  expect(edited.modelNeedsResolution).toBe(false);
});
