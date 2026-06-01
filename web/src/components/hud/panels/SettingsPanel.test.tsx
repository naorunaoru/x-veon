import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { SettingsPanel } from './SettingsPanel';
import { useAppStore } from '@/store';
import type { QueuedFile } from '@/store';

vi.mock('@/pipeline/inference', () => ({
  getAvailableSizes: () => new Set(['S', 'M', 'L']),
  switchModelSize: vi.fn().mockResolvedValue(undefined),
}));
vi.mock('@/hooks/useProcessFile', () => ({
  useProcessFile: () => ({ processFile: vi.fn(), isProcessing: false }),
}));

function makeFile(): QueuedFile {
  return {
    id: 'a', file: null, name: 'a', originalName: 'a.raf', thumbnailUrl: null,
    metadata: null, cfaType: 'xtrans', status: 'done', error: null, progress: null,
    result: null, resultMethod: null, lensProfile: null, lookPreset: 'default',
    openDrtOverrides: {}, preProcessOverrides: {},
  };
}

describe('SettingsPanel', () => {
  beforeEach(() => useAppStore.setState({
    openPanel: 'settings', files: [makeFile()], selectedFileId: 'a',
    demosaicMethod: 'neural-net', modelSize: 'S',
  }));

  it('changes the demosaic method', () => {
    const spy = vi.spyOn(useAppStore.getState(), 'setDemosaicMethod');
    render(<SettingsPanel />);
    fireEvent.change(screen.getByLabelText('Demosaic method'), { target: { value: 'bilinear' } });
    expect(spy).toHaveBeenCalledWith('bilinear');
  });

  it('selects a model size', () => {
    const spy = vi.spyOn(useAppStore.getState(), 'setModelSize');
    render(<SettingsPanel />);
    screen.getByRole('button', { name: 'M' }).click();
    expect(spy).toHaveBeenCalledWith('M');
  });
});
