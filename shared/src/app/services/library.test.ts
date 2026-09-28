import { beforeEach, expect, it, vi } from 'vitest';
import { useAppStore } from '@/app/store';
import { fakeHost, fakePhoto } from '@/test/fake-host';
import { setHost } from './host';
import { importFiles, removeFile, startLibraryWatching } from './library';
import { fromLibraryPhoto } from '@/app/store/photo';
vi.mock('./processing', () => ({ discardResult: vi.fn() }));
vi.mock('./persistence', () => ({
  cancelPhotoSave: vi.fn(async () => {}),
  pausePersistence: vi.fn(),
  resumePersistence: vi.fn(),
}));
vi.mock('@/app/lens/lensfun', () => ({ matchLens: vi.fn(async () => ({ lensModel: 'XF35' })) }));
import { matchLens } from '@/app/lens/lensfun';
import { discardResult } from './processing';
import { cancelPhotoSave } from './persistence';
let host: ReturnType<typeof fakeHost>;
beforeEach(() => {
  vi.clearAllMocks();
  host = fakeHost();
  setHost(host);
  useAppStore.setState({ files: [], selectedFileId: null });
});
it('adds host snapshots, selects the first added photo and matches its lens', async () => {
  const photo = fakePhoto();
  photo.facts.metadata = { camera: 'Fuji', lensModel: 'XF35', focalLength: 35, fNumber: 2 };
  vi.mocked(host.library.addFiles).mockResolvedValue({ photos: [photo], selectedIds: ['a'], complete: true });
  await importFiles([new File(['x'], 'a.raf')]);
  expect(useAppStore.getState().selectedFileId).toBe('a');
  await vi.waitFor(() =>
    expect(useAppStore.getState().files[0].lensProfile).toMatchObject({ lensModel: 'XF35' }),
  );
  expect(matchLens).toHaveBeenCalledWith('Fuji', 'XF35');
});
it('removes only when the host supports it, draining saves before host deletion', async () => {
  useAppStore.setState({ files: [fromLibraryPhoto(fakePhoto())], selectedFileId: 'a' });
  await removeFile('a');
  expect(useAppStore.getState().files).toHaveLength(1);
  host.library.remove = vi.fn(async () => {});
  await removeFile('a');
  expect(useAppStore.getState().files).toEqual([]);
  expect(discardResult).toHaveBeenCalledWith('a');
  expect(cancelPhotoSave).toHaveBeenCalledWith('a');
  expect(host.library.remove).toHaveBeenCalledWith('a');
  expect(vi.mocked(cancelPhotoSave).mock.invocationCallOrder[0]).toBeLessThan(
    vi.mocked(host.library.remove).mock.invocationCallOrder[0],
  );
});
it('applies thumbnail facts without replacing a local edit and releases its subscription', () => {
  let publish!: Parameters<NonNullable<typeof host.library.onChange>>[0];
  const stop = vi.fn();
  host.library.onChange = (listener) => {
    publish = listener;
    return stop;
  };
  useAppStore.setState({ files: [fromLibraryPhoto(fakePhoto())] });
  const unsubscribe = startLibraryWatching();
  useAppStore.getState().setFileLookPreset('a', 'umbra');
  publish({
    kind: 'facts',
    snapshot: { complete: true, photos: [{ ...fakePhoto(), thumbnailUrl: 'blob:thumb' }] },
  });
  expect(useAppStore.getState().files[0]).toMatchObject({
    thumbnailUrl: 'blob:thumb',
    edit: { lookPreset: 'umbra' },
  });
  unsubscribe();
  expect(stop).toHaveBeenCalledOnce();
});
