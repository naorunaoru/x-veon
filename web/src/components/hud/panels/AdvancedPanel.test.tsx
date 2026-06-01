import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { AdvancedPanel } from './AdvancedPanel';
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

describe('AdvancedPanel', () => {
  beforeEach(() => useAppStore.setState({ openPanel: 'advanced', files: [makeFile()], selectedFileId: 'a' }));

  it('renders the group headers (collapsed by default)', () => {
    render(<AdvancedPanel />);
    ['Tonescale', 'Purity', 'Brilliance C/M/Y', 'Hue Shift', 'Detail', 'Output'].forEach((g) =>
      expect(screen.getByRole('button', { name: new RegExp(g) })).toBeInTheDocument());
  });

  it('expanding a group reveals its sliders', () => {
    render(<AdvancedPanel />);
    fireEvent.click(screen.getByRole('button', { name: /Output/ }));
    expect(screen.getByText('Peak Luminance')).toBeInTheDocument();
  });

  it('toggling a sub-section enable writes the enable override', () => {
    const spy = vi.spyOn(useAppStore.getState(), 'setFileOpenDrtOverride');
    render(<AdvancedPanel />);
    fireEvent.click(screen.getByRole('button', { name: /Hue Shift/ }));
    // The 'default' look enables hs_rgb_enable, so toggling the RGB switch sends false.
    fireEvent.click(screen.getByRole('switch', { name: 'RGB' }));
    expect(spy).toHaveBeenCalledWith('a', 'hs_rgb_enable', false);
  });
});
