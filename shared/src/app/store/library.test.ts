import { expect, it } from 'vitest';
import { useAppStore, type QueuedFile } from '@/app/store';
import { fromLibraryPhoto } from './photo';
import { fakePhoto } from '@/test/fake-host';
it('advances through removal, falls back to the previous last photo, and handles an empty library', () => {
  useAppStore.setState({ files: ['a', 'b', 'c', 'd'].map(id => ({ id }) as QueuedFile), currentId: 'b' });
  useAppStore.getState().removeFile('b');
  expect(useAppStore.getState().currentId).toBe('c');
  useAppStore.getState().removeFile('a');
  expect(useAppStore.getState().currentId).toBe('c');
  useAppStore.getState().removeFile('c');
  expect(useAppStore.getState().currentId).toBe('d');
  useAppStore.getState().removeFile('d');
  expect(useAppStore.getState().currentId).toBeNull();
  useAppStore.setState({ files: ['a', 'b'].map(id => ({ id }) as QueuedFile), currentId: 'b' });
  useAppStore.getState().removeFile('b');
  expect(useAppStore.getState().currentId).toBe('a');
});
it('restores settings for an empty library and ignores a selection whose photo is missing', () => {
  useAppStore.setState({ files: [], currentId: null });
  useAppStore.getState().restoreFromDb([], { demosaicMethod: 'dht', currentId: 'gone' });
  expect(useAppStore.getState()).toMatchObject({ currentId: null, demosaicMethod: 'dht' });
  useAppStore.getState().restoreFromDb([{ id: 'a' } as QueuedFile], { currentId: 'gone' });
  expect(useAppStore.getState().currentId).toBe('a');
});
it('keeps current settings that were never saved', () => {
  useAppStore.setState({ files: [], currentId: null, demosaicMethod: 'bilinear', exportFormat: 'tiff', exportQuality: 80 });
  useAppStore.getState().restoreFromDb([], {});
  expect(useAppStore.getState()).toMatchObject({ demosaicMethod: 'bilinear', exportFormat: 'tiff', exportQuality: 80 });
});
it('setFileRating bumps the revision and marks the session', () => {
  const file = fromLibraryPhoto(fakePhoto());
  useAppStore.setState({ files: [file], currentId: 'a' });
  useAppStore.getState().setFileRating('a', 3);
  const rated = useAppStore.getState().files[0];
  expect(rated).toMatchObject({ rating: 3, editing: 'session' });
  expect(rated.editRevision).toBeGreaterThan(file.editRevision);
  expect(rated.edit).toBe(file.edit);
});
it('setFileRating on a view-only photo changes only the rating', () => {
  const file: QueuedFile = { ...fromLibraryPhoto(fakePhoto()), editing: 'view-only', editingNote: 'newer schema' };
  useAppStore.setState({ files: [file], currentId: 'a' });
  useAppStore.getState().setFileRating('a', -1);
  expect(useAppStore.getState().files[0]).toEqual({ ...file, rating: -1 });
});
it('setFileRating with the same rating does nothing', () => {
  const file = fromLibraryPhoto(fakePhoto());
  useAppStore.setState({ files: [file], currentId: 'a' });
  useAppStore.getState().setFileRating('a', 0);
  expect(useAppStore.getState().files[0]).toBe(file);
});
