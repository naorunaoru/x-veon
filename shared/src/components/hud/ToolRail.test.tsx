import { fromLibraryPhoto } from '@/app/store/photo';
import { fakePhoto, defaultEdit } from '@/test/fake-host';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ToolRail } from './ToolRail';
import { useAppStore } from '@/app/store';
import type { QueuedFile } from '@/app/store';

function fileWith(over: Partial<QueuedFile>): QueuedFile {
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
    ...over,
  };
}

describe('ToolRail', () => {
  beforeEach(() => useAppStore.setState({ openPanel: null, files: [fileWith({})], selectedFileId: 'a' }));

  it('renders the rail buttons', () => {
    render(<ToolRail />);
    ['Exposure', 'White balance', 'Rendering', 'Detail', 'Settings'].forEach((label) => {
      expect(screen.getByRole('button', { name: label })).toBeInTheDocument();
    });
  });

  it('opens the adjustment panel on click', () => {
    const spy = vi.spyOn(useAppStore.getState(), 'setOpenPanel');
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
    useAppStore.setState({
      files: [fileWith({ edit: { ...defaultEdit(), preProcessOverrides: { exposure: 0.5 } } })],
      selectedFileId: 'a',
    });
    const { container } = render(<ToolRail />);
    expect(container.querySelector('.xv-rail-btn__dot')).not.toBeNull();
  });
});
