import { expect, it } from 'vitest';
import { useAppStore, type QueuedFile } from '@/app/store';
it('advances through removal, falls back to the previous last photo, and handles an empty library', () => {
  useAppStore.setState({ files: ['a', 'b', 'c', 'd'].map(id => ({ id }) as QueuedFile), selectedFileId: 'b' });
  useAppStore.getState().removeFile('b');
  expect(useAppStore.getState().selectedFileId).toBe('c');
  useAppStore.getState().removeFile('a');
  expect(useAppStore.getState().selectedFileId).toBe('c');
  useAppStore.getState().removeFile('c');
  expect(useAppStore.getState().selectedFileId).toBe('d');
  useAppStore.getState().removeFile('d');
  expect(useAppStore.getState().selectedFileId).toBeNull();
  useAppStore.setState({ files: ['a', 'b'].map(id => ({ id }) as QueuedFile), selectedFileId: 'b' });
  useAppStore.getState().removeFile('b');
  expect(useAppStore.getState().selectedFileId).toBe('a');
});
