import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { LooksPanel } from './LooksPanel';
import { useAppStore } from '@/store';
import type { QueuedFile } from '@/store';

function makeFile(): QueuedFile {
  return {
    id: 'a', file: null, name: 'a', originalName: 'a.raf', thumbnailUrl: null,
    metadata: null, cfaType: 'xtrans', status: 'done', error: null, progress: null,
    result: null, resultMethod: null, lensProfile: null, lookPreset: 'default',
    openDrtOverrides: {}, preProcessOverrides: {},
  };
}

describe('LooksPanel', () => {
  beforeEach(() => useAppStore.setState({ openPanel: 'looks', files: [makeFile()], selectedFileId: 'a' }));

  it('renders the five looks and marks the active one', () => {
    render(<LooksPanel />);
    ['Default', 'Colorful', 'Umbra', 'Base', 'Flat'].forEach((l) =>
      expect(screen.getByText(l)).toBeInTheDocument());
    expect(screen.getByText('Default').closest('.xv-card')!.className).toMatch(/is-selected/);
  });

  it('selecting a look calls setFileLookPreset', () => {
    const spy = vi.spyOn(useAppStore.getState(), 'setFileLookPreset');
    render(<LooksPanel />);
    screen.getByText('Colorful').click();
    expect(spy).toHaveBeenCalledWith('a', 'colorful');
  });
});
