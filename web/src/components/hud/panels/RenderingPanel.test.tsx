import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { RenderingPanel } from './RenderingPanel';
import { useAppStore } from '@/app/store';
import type { QueuedFile } from '@/app/store';

function makeFile(): QueuedFile {
  return {
    id: 'a', file: null, name: 'a', originalName: 'a.raf', thumbnailUrl: null,
    metadata: null, cfaType: 'xtrans', status: 'done', error: null, progress: null,
    result: null, resultMethod: null, lensProfile: null, lookPreset: 'default',
    openDrtOverrides: {}, preProcessOverrides: {},
  };
}

describe('RenderingPanel', () => {
  beforeEach(() => useAppStore.setState({ openPanel: 'advanced', files: [makeFile()], selectedFileId: 'a' }));

  it('renders the header and the five section labels', () => {
    render(<RenderingPanel />);
    expect(screen.getByText('Rendering')).toBeInTheDocument();
    expect(screen.getByText('OpenDRT')).toBeInTheDocument();
    ['Tonescale', 'Color rendering', 'Whites', 'Expert'].forEach((s) =>
      expect(screen.getByText(s)).toBeInTheDocument());
    // "Purity" is both a section title and a wheel mode — at least one present.
    expect(screen.getAllByText('Purity').length).toBeGreaterThan(0);
  });

  it('selecting a base look sets the file look preset', () => {
    const spy = vi.spyOn(useAppStore.getState(), 'setFileLookPreset');
    render(<RenderingPanel />);
    fireEvent.click(screen.getByRole('button', { name: 'Colorful' }));
    expect(spy).toHaveBeenCalledWith('a', 'colorful');
  });

  it('a tonescale preset chip writes the preset overrides', () => {
    const spy = vi.spyOn(useAppStore.getState(), 'setFileOpenDrtOverride');
    render(<RenderingPanel />);
    fireEvent.click(screen.getByRole('button', { name: 'High' }));
    // 'high-contrast' enables low contrast.
    expect(spy).toHaveBeenCalledWith('a', 'tn_lcon_enable', true);
  });

  it('the colour wheel exposes the three layer modes', () => {
    render(<RenderingPanel />);
    ['Brilliance', 'Hue twist', 'Purity'].forEach((m) =>
      expect(screen.getByRole('button', { name: m })).toBeInTheDocument());
  });

  it('the Expert drawer reveals raw OpenDRT keys when opened', () => {
    render(<RenderingPanel />);
    expect(screen.queryByText('peak_luminance')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Expert/ }));
    expect(screen.getByText('peak_luminance')).toBeInTheDocument();
    expect(screen.getByText('tn_lg')).toBeInTheDocument();
  });
});
