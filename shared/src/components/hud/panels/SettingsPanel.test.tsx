import { fromLibraryPhoto } from '@/app/store/photo';
import { fakePhoto, defaultEdit } from '@/test/fake-host';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { SettingsPanel } from './SettingsPanel';
import { useAppStore } from '@/app/store';
import type { QueuedFile } from '@/app/store';

vi.mock('@/app/hooks/useModelSizes', () => ({
  useModelSizes: () => ({
    available: new Set(['S', 'M', 'L']),
    switchTo: vi.fn().mockResolvedValue(undefined),
  }),
}));
vi.mock('@/app/hooks/useProcessing', () => ({
  useProcessing: () => ({ processFile: vi.fn(), isProcessing: false }),
}));

function makeFile(): QueuedFile {
  return {
    ...fromLibraryPhoto(fakePhoto()),
    id: 'a',
    name: 'a',
    originalName: 'a.raf',
    thumbnailUrl: null,
    metadata: null,
    cfaType: 'xtrans',
    status: 'done',
    error: null,
    progress: null,
    result: null,
    resultMethod: null,
    lensProfile: null,
    edit: { ...defaultEdit(), lookPreset: 'default', openDrtOverrides: {}, preProcessOverrides: {} },
  };
}

describe('SettingsPanel', () => {
  beforeEach(() =>
    useAppStore.setState({
      openPanel: 'settings',
      files: [makeFile()],
      selectedFileId: 'a',
      demosaicMethod: 'neural-net',
      modelSize: 'S',
    }),
  );

  it('changes the demosaic method', () => {
    const spy = vi.spyOn(useAppStore.getState(), 'setDemosaicMethod');
    render(<SettingsPanel />);
    fireEvent.change(screen.getByLabelText('Default demosaic method'), { target: { value: 'bilinear' } });
    expect(spy).toHaveBeenCalledWith('bilinear');
  });

  it('selects a model size', () => {
    const spy = vi.spyOn(useAppStore.getState(), 'setModelSize');
    render(<SettingsPanel />);
    fireEvent.change(screen.getByLabelText('Default model'), { target: { value: 'M' } });
    expect(spy).toHaveBeenCalledWith('M');
  });

  it('shows the dev build stamp without a channel link', () => {
    render(<SettingsPanel />);
    expect(screen.getByTestId('xv-build')).toHaveTextContent('Dev');
    expect(screen.getByTestId('xv-build')).toHaveTextContent('test');
    expect(screen.queryByRole('link', { name: /^Open / })).toBeNull();
  });
});
