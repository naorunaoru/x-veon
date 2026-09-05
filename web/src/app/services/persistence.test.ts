import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const idb = vi.hoisted(() => ({
  putSetting: vi.fn().mockResolvedValue(undefined), debouncedPutFile: vi.fn(), putFile: vi.fn(),
  getAllFiles: vi.fn(), getSetting: vi.fn(), cancelPendingPut: vi.fn(),
}));
vi.mock('@/app/storage/idb-storage', () => idb);
vi.mock('@/app/storage/opfs-storage', () => ({ readThumbnail: vi.fn().mockResolvedValue(null) }));

import { useAppStore } from '@/app/store';
import type { QueuedFile } from '@/app/store';
import { startPersistence, restore, fileToPersistedFile } from './persistence';
import { configFromPreset, configWithOverrides, TONESCALE_PRESETS } from '@/renderer/grading/opendrt-params';

function makeFile(id: string): QueuedFile {
  return {
    id, file: null, name: id, originalName: `${id}.raf`, thumbnailUrl: null,
    metadata: null, cfaType: 'xtrans', status: 'done', error: null, progress: null,
    result: null, resultMethod: null, lensProfile: null, lookPreset: 'default',
    openDrtOverrides: {}, preProcessOverrides: {},
  };
}

describe('persistence service', () => {
  let stop: () => void;
  beforeEach(() => {
    vi.clearAllMocks();
    useAppStore.setState({ files: [makeFile('a')], selectedFileId: 'a', exportQuality: 95, demosaicMethod: 'neural-net' });
    stop = startPersistence();
  });
  afterEach(() => stop());

  it('writes a changed file record exactly once, debounced', () => {
    useAppStore.getState().setFileLookPreset('a', 'colorful');
    expect(idb.debouncedPutFile).toHaveBeenCalledTimes(1);
    expect(idb.debouncedPutFile.mock.calls[0][0]).toMatchObject({ id: 'a', lookPreset: 'colorful' });
    expect(idb.putFile).not.toHaveBeenCalled();
  });

  it('writes settings keys as they change and nothing else', () => {
    useAppStore.getState().setExportQuality(80);
    expect(idb.putSetting).toHaveBeenCalledWith('exportQuality', 80);
    expect(idb.putSetting).toHaveBeenCalledTimes(1);
    useAppStore.getState().selectFile(null);
    expect(idb.putSetting).toHaveBeenCalledWith('selectedFileId', null);
  });

  it('does not write while a file is processing; the error record is written', () => {
    useAppStore.getState().updateFileStatus('a', 'processing');
    expect(idb.debouncedPutFile).not.toHaveBeenCalled();
    useAppStore.getState().updateFileStatus('a', 'error', 'boom');
    expect(idb.debouncedPutFile).toHaveBeenCalledTimes(1);
    expect(idb.debouncedPutFile.mock.calls[0][0]).toMatchObject({ status: 'error', error: 'boom' });
  });

  it('restores files and settings from storage', async () => {
    idb.getAllFiles.mockResolvedValue([{
      id: 'r', name: 'r', originalName: 'r.raf', fileSize: 1, cfaType: 'bayer', camera: 'Sony', lensModel: null,
      focalLength: null, fNumber: null, status: 'queued', error: null, resultMethod: null, resultMeta: null,
      cachedMethods: [], lookPreset: 'default', lensProfile: null, openDrtOverrides: {}, preProcessOverrides: {}, addedAt: 1,
    }]);
    idb.getSetting.mockImplementation(async (key: string) => ({ exportQuality: 80, selectedFileId: 'r' } as Record<string, unknown>)[key]);
    const restored = await restore();
    expect(restored.files[0]).toMatchObject({ id: 'r', status: 'queued', metadata: { camera: 'Sony', lensModel: '' } });
    expect(restored.settings).toEqual({ demosaicMethod: undefined, exportFormat: undefined, exportQuality: 80, selectedFileId: 'r' });
  });
  it('does not persist view-only updates and stops writing after unsubscribe', () => {
    useAppStore.getState().setViewScale(2);
    useAppStore.getState().setOpenPanel('scopes');
    expect(idb.putSetting).not.toHaveBeenCalled();
    expect(idb.debouncedPutFile).not.toHaveBeenCalled();
    stop();
    useAppStore.getState().setExportQuality(71);
    useAppStore.getState().setFileLookPreset('a', 'base');
    expect(idb.putSetting).not.toHaveBeenCalled();
    expect(idb.debouncedPutFile).not.toHaveBeenCalled();
  });

  it.each(['colorful', 'marvelous'] as const)('round-trips %s with its edits and appearance intact', async (lookPreset) => {
    const original: QueuedFile = {
      ...makeFile('r'), lookPreset,
      openDrtOverrides: { ...TONESCALE_PRESETS['aces-2'].overrides, cwp: 0.4 },
      preProcessOverrides: { exposure: 1, wb_temp: 0.2, sharpen_amount: 0.5 },
      lookHistory: [{ lookPreset: 'umbra', openDrtOverrides: {} }],
    };
    const saved = fileToPersistedFile(original);
    expect(saved).not.toHaveProperty('lookHistory');
    idb.getAllFiles.mockResolvedValue([saved]);
    idb.getSetting.mockResolvedValue(undefined);
    const { files: [restored] } = await restore();
    expect(restored).toMatchObject({ lookPreset, openDrtOverrides: original.openDrtOverrides, preProcessOverrides: original.preProcessOverrides });
    expect(configWithOverrides(configFromPreset(restored.lookPreset), restored.openDrtOverrides, restored.preProcessOverrides))
      .toEqual(configWithOverrides(configFromPreset(original.lookPreset), original.openDrtOverrides, original.preProcessOverrides));
  });

});
