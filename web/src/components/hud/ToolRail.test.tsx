import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ToolRail } from './ToolRail';
import { useAppStore } from '@/store';
import type { QueuedFile } from '@/store';

function fileWith(over: Partial<QueuedFile>): QueuedFile {
  return {
    id: 'a', file: null, name: 'a', originalName: 'a.raf', thumbnailUrl: null,
    metadata: null, cfaType: 'xtrans', status: 'done', error: null, progress: null,
    result: null, resultMethod: null, lensProfile: null, lookPreset: 'default',
    openDrtOverrides: {}, preProcessOverrides: {}, ...over,
  };
}

describe('ToolRail', () => {
  beforeEach(() => useAppStore.setState({ openPanel: null, files: [fileWith({})], selectedFileId: 'a' }));

  it('renders the rail buttons', () => {
    render(<ToolRail />);
    ['Scopes', 'Exposure', 'White balance', 'Rendering', 'Detail', 'Settings'].forEach((label) => {
      expect(screen.getByRole('button', { name: label })).toBeInTheDocument();
    });
  });

  it('toggles a panel open on click', () => {
    const spy = vi.spyOn(useAppStore.getState(), 'togglePanel');
    render(<ToolRail />);
    screen.getByRole('button', { name: 'Exposure' }).click();
    expect(spy).toHaveBeenCalledWith('exposure');
  });

  it('marks the active button pressed', () => {
    useAppStore.setState({ openPanel: 'advanced' });
    render(<ToolRail />);
    expect(screen.getByRole('button', { name: 'Rendering' })).toHaveAttribute('aria-pressed', 'true');
  });

  it('shows a modified dot on a section with overrides', () => {
    useAppStore.setState({ files: [fileWith({ preProcessOverrides: { exposure: 0.5 } })], selectedFileId: 'a' });
    const { container } = render(<ToolRail />);
    expect(container.querySelector('.xv-rail-btn__dot')).not.toBeNull();
  });
});
