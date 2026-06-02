import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { DetailPanel } from './DetailPanel';
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

describe('DetailPanel', () => {
  beforeEach(() => useAppStore.setState({ openPanel: 'detail', files: [makeFile()], selectedFileId: 'a' }));

  it('renders only the sharpening control', () => {
    render(<DetailPanel />);
    expect(screen.getByText('Detail')).toBeInTheDocument();
    expect(screen.getByText('Sharpening')).toBeInTheDocument();
  });

  it('changing sharpening writes the sharpen_amount pre-override', () => {
    const spy = vi.spyOn(useAppStore.getState(), 'setFilePreProcessOverride');
    render(<DetailPanel />);
    const slider = screen.getByRole('slider', { name: 'Sharpening' });
    fireEvent.keyDown(slider, { key: 'ArrowRight' });
    expect(spy).toHaveBeenCalledWith('a', 'sharpen_amount', expect.any(Number));
  });
});
