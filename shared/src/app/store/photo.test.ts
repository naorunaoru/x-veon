import { expect, it } from 'vitest';
import { fakePhoto } from '@/test/fake-host';
import { editPhoto, fromLibraryPhoto, processingKey } from './photo';
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
  const edited = editPhoto(file, { preProcessOverrides: { exposure: 1 } });
  expect(edited.edit.model).toEqual(actual);
  expect(edited.processedKey).toBe(processingKey(edited, { demosaicMethod: 'neural-net', modelSize: 'S' }));
});
it('keeps a real method or model change pending for reprocessing', () => {
  const file = processedPhoto();
  for (const patch of [
    { demosaicMethod: 'dht' as const },
    { model: { size: 'M' as const, sha256: 'new' } },
  ]) {
    const edited = editPhoto(file, patch);
    expect(edited.processedKey).not.toBe(
      processingKey(edited, { demosaicMethod: 'neural-net', modelSize: 'S' }),
    );
  }
});
it('does not acknowledge a result from an earlier requested model', () => {
  const file = processedPhoto();
  file.processedKey = 'neural-net:S:earlier';
  const edited = editPhoto(file, { preProcessOverrides: { exposure: 1 } });
  expect(edited.processedKey).toBe('neural-net:S:earlier');
});
