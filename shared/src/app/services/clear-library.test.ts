import { beforeEach, expect, it, vi } from 'vitest';
import { fakeHost, fakePhoto } from '@/test/fake-host';
import { fromLibraryPhoto } from '@/app/store/photo';
import { setHost } from './host';
import { useAppStore } from '@/app/store';
vi.mock('./persistence', () => ({
  pausePersistence: vi.fn(async () => {}),
  resumePersistence: vi.fn(),
  cancelPhotoSave: vi.fn(),
  restore: vi.fn(() => new Promise(() => {})),
}));
vi.mock('./processing', () => ({ discardResult: vi.fn() }));
import { pausePersistence, resumePersistence, restore } from './persistence';
import { clearLibrary, importFiles } from './library';
beforeEach(() => vi.clearAllMocks());
it('surfaces a blocked clear promptly and keeps writes suspended until retry succeeds', async () => {
  const host = fakeHost();
  host.library.clear = vi
    .fn()
    .mockRejectedValueOnce(new Error('Close other tabs'))
    .mockResolvedValue(undefined);
  setHost(host);
  useAppStore.setState({ files: [fromLibraryPhoto(fakePhoto())], selectedFileId: 'a' });
  const failed = clearLibrary();
  // Rejection must not wait for a fresh DB open behind an uncancellable blocked delete.
  const outcome = await Promise.race([
    failed.then(
      () => 'resolved',
      (error) => error.message,
    ),
    new Promise((resolve) => setTimeout(() => resolve('stalled'), 30)),
  ]);
  expect(outcome).toBe('Close other tabs');
  expect(restore).not.toHaveBeenCalled();
  expect(resumePersistence).not.toHaveBeenCalled();
  await importFiles([new File(['x'], 'a.raf')]);
  expect(host.library.addFiles).not.toHaveBeenCalled();
  await clearLibrary();
  expect(pausePersistence).toHaveBeenCalledTimes(2);
  expect(resumePersistence).toHaveBeenCalledOnce();
});
