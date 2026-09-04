import { describe, it, expect, beforeEach } from 'vitest';
import { useAppStore } from '@/app/store';
import type { QueuedFile } from '@/app/store';

function makeFile(id: string): QueuedFile {
  return {
    id, file: null, name: id, originalName: `${id}.raf`, thumbnailUrl: null,
    metadata: null, cfaType: 'xtrans', status: 'done', error: null, progress: null,
    result: null, resultMethod: null, lensProfile: null, lookPreset: 'default',
    openDrtOverrides: { tn_con: 1.0, brl_r: 0.2 }, preProcessOverrides: { exposure: 1.5, wb_temp: 0.3 },
  };
}

describe('openPanel', () => {
  beforeEach(() => useAppStore.setState({ openPanel: null }));

  it('setOpenPanel sets and clears the open panel', () => {
    useAppStore.getState().setOpenPanel('exposure');
    expect(useAppStore.getState().openPanel).toBe('exposure');
    useAppStore.getState().setOpenPanel(null);
    expect(useAppStore.getState().openPanel).toBeNull();
  });

  it('togglePanel opens, then closes the same panel, then switches', () => {
    const { togglePanel } = useAppStore.getState();
    togglePanel('exposure');
    expect(useAppStore.getState().openPanel).toBe('exposure');
    togglePanel('exposure');
    expect(useAppStore.getState().openPanel).toBeNull();
    togglePanel('scopes');
    togglePanel('settings');
    expect(useAppStore.getState().openPanel).toBe('settings');
  });
});

describe('per-section override clears', () => {
  beforeEach(() => useAppStore.setState({ files: [makeFile('a')], selectedFileId: 'a' }));

  it('clearFileOpenDrtOverrides removes only the named keys', () => {
    useAppStore.getState().clearFileOpenDrtOverrides('a', ['tn_con']);
    const f = useAppStore.getState().files.find((x) => x.id === 'a')!;
    expect('tn_con' in f.openDrtOverrides).toBe(false);
    expect(f.openDrtOverrides.brl_r).toBe(0.2);
  });

  it('clearFilePreProcessOverrides removes only the named keys', () => {
    useAppStore.getState().clearFilePreProcessOverrides('a', ['wb_temp']);
    const f = useAppStore.getState().files.find((x) => x.id === 'a')!;
    expect('wb_temp' in f.preProcessOverrides).toBe(false);
    expect(f.preProcessOverrides.exposure).toBe(1.5);
  });
});
