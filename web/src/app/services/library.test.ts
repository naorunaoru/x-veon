import { beforeEach, describe, expect, it, vi } from 'vitest';

const opfs = vi.hoisted(() => ({
  writeRaw: vi.fn().mockResolvedValue(undefined), writeThumbnail: vi.fn().mockResolvedValue(undefined),
  deleteAllForFile: vi.fn().mockResolvedValue(undefined), listRawFileIds: vi.fn().mockResolvedValue(new Set(['a', 'zombie'])),
}));
vi.mock('@/app/storage/opfs-storage', () => opfs);
const idb = vi.hoisted(() => ({
  deleteFile: vi.fn().mockResolvedValue(undefined), cancelPendingPut: vi.fn(),
  // still called by the monolithic store's actions until Task 9
  putFile: vi.fn().mockResolvedValue(undefined), putSetting: vi.fn().mockResolvedValue(undefined), debouncedPutFile: vi.fn(),
}));
vi.mock('@/app/storage/idb-storage', () => idb);
vi.mock('@/pipeline/decode/raf-thumbnail', () => ({
  extractRafThumbnail: () => new Blob(['t']),
  extractRafQuickMetadata: () => ({ camera: 'Fuji X-T10', lensModel: 'XF35', focalLength: 35, fNumber: 2 }),
}));
const matchLens = vi.fn().mockResolvedValue({ lensModel: 'XF35', mount: 'X', cropfactor: 1.5, distortion: [], tca: [], vignetting: [] });
vi.mock('@/app/lens/lensfun', () => ({ matchLens: (...args: unknown[]) => matchLens(...args) }));
const discardResult = vi.fn();
vi.mock('@/app/services/processing', () => ({ discardResult: (...args: unknown[]) => discardResult(...args) }));

import { useAppStore } from '@/app/store';
import { importFiles, removeFile, cleanupOrphans } from './library';

// jsdom's File may lack arrayBuffer(); the service only needs it to resolve to bytes.
if (typeof File.prototype.arrayBuffer !== 'function') {
  Object.defineProperty(File.prototype, 'arrayBuffer', { value: () => Promise.resolve(new ArrayBuffer(4)) });
}

describe('library service', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useAppStore.setState({ files: [], selectedFileId: null });
    globalThis.URL.createObjectURL = vi.fn(() => 'blob:thumb');
    globalThis.URL.revokeObjectURL = vi.fn();
  });

  it('imports RAW files only, selects the first, and writes bytes, thumbnail and metadata', async () => {
    importFiles([new File(['x'], 'one.RAF'), new File(['y'], 'notes.txt'), new File(['z'], 'two.arw')]);
    const { files, selectedFileId } = useAppStore.getState();
    expect(files.map((f) => f.originalName)).toEqual(['one.RAF', 'two.arw']);
    expect(files.map((f) => f.cfaType)).toEqual(['xtrans', 'bayer']);
    expect(selectedFileId).toBe(files[0].id);

    await vi.waitFor(() => expect(useAppStore.getState().files[0].lensProfile).not.toBeNull());
    expect(opfs.writeRaw).toHaveBeenCalledTimes(2);
    expect(opfs.writeThumbnail).toHaveBeenCalledTimes(2);
    expect(useAppStore.getState().files[0]).toMatchObject({ thumbnailUrl: 'blob:thumb', metadata: { camera: 'Fuji X-T10' } });
    expect(matchLens).toHaveBeenCalledWith('Fuji X-T10', 'XF35');
  });

  it('removes a file from the store, both storages, the pending write and the processing service', () => {
    importFiles([new File(['x'], 'a.raf')]);
    const id = useAppStore.getState().files[0].id;
    removeFile(id);
    expect(useAppStore.getState().files).toEqual([]);
    expect(opfs.deleteAllForFile).toHaveBeenCalledWith(id);
    expect(idb.deleteFile).toHaveBeenCalledWith(id);
    expect(idb.cancelPendingPut).toHaveBeenCalledWith(id);
    expect(discardResult).toHaveBeenCalledWith(id);
  });

  it('deletes OPFS entries that are not in the library', async () => {
    await cleanupOrphans(new Set(['a']));
    expect(opfs.deleteAllForFile).toHaveBeenCalledWith('zombie');
    expect(opfs.deleteAllForFile).not.toHaveBeenCalledWith('a');
  });
  it('does not write files removed before their bytes arrive', async () => {
    let release!: (bytes: ArrayBuffer) => void;
    const file = new File(['raw'], 'late.raf');
    Object.defineProperty(file, 'arrayBuffer', { value: () => new Promise<ArrayBuffer>(resolve => { release = resolve; }) });
    importFiles([file]);
    removeFile(useAppStore.getState().files[0].id);
    release(new ArrayBuffer(4));
    await Promise.resolve(); await Promise.resolve();
    expect(opfs.writeRaw).not.toHaveBeenCalled();
    expect(URL.createObjectURL).not.toHaveBeenCalled();
  });
  it('cleans a write that finishes after removal and releases the thumbnail URL', async () => {
    let finish!: () => void;
    opfs.writeRaw.mockImplementationOnce(() => new Promise<void>(resolve => { finish = resolve; }));
    importFiles([new File(['raw'], 'pending.raf')]);
    const id = useAppStore.getState().files[0].id;
    await vi.waitFor(() => expect(opfs.writeRaw).toHaveBeenCalled());
    removeFile(id);
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:thumb');
    finish();
    await vi.waitFor(() => expect(opfs.deleteAllForFile).toHaveBeenCalledTimes(2));
    expect(useAppStore.getState().files).toEqual([]);
  });

});
