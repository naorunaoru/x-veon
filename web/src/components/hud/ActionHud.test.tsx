import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ActionHud } from './ActionHud';
import { useAppStore } from '@/app/store';
import type { QueuedFile } from '@/app/store';

// ActionHud calls useProcessing()/useExport(), whose modules import the WASM
// pipeline. Mock them so the test renders without WASM; the real store still
// drives the button disabled states.
vi.mock('@/app/hooks/useProcessing', () => ({
  useProcessing: () => ({ processFile: vi.fn(), isProcessing: false }),
}));
vi.mock('@/app/hooks/useExport', () => ({
  useExport: () => ({ exportFile: vi.fn(), isExporting: false }),
}));

function makeFile(id: string, status: QueuedFile['status']): QueuedFile {
  return {
    id, file: null, name: id, originalName: `${id}.raf`, thumbnailUrl: null,
    metadata: null, cfaType: 'xtrans', status, error: null, progress: null,
    result: null, resultMethod: null, lensProfile: null, lookPreset: 'default',
    openDrtOverrides: {}, preProcessOverrides: {},
  };
}

describe('ActionHud', () => {
  beforeEach(() => {
    useAppStore.setState({ initialized: true, files: [], selectedFileId: null });
  });

  it('disables Process when nothing is selected', () => {
    render(<ActionHud />);
    expect(screen.getByRole('button', { name: 'Process' })).toBeDisabled();
  });

  it('enables Process for a selected, initialized file', () => {
    useAppStore.setState({ files: [makeFile('a', 'queued')], selectedFileId: 'a' });
    render(<ActionHud />);
    expect(screen.getByRole('button', { name: 'Process' })).not.toBeDisabled();
  });

  it('disables Export until the file is done', () => {
    useAppStore.setState({ files: [makeFile('a', 'queued')], selectedFileId: 'a' });
    render(<ActionHud />);
    expect(screen.getByRole('button', { name: 'Export' })).toBeDisabled();
  });

  it('enables Export when the file is done', () => {
    useAppStore.setState({ files: [makeFile('a', 'done')], selectedFileId: 'a' });
    render(<ActionHud />);
    expect(screen.getByRole('button', { name: 'Export' })).not.toBeDisabled();
  });
});
