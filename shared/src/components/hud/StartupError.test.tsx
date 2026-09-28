import { fromLibraryPhoto } from '@/app/store/photo';
import { fakePhoto } from '@/test/fake-host';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { useAppStore, type QueuedFile } from '@/app/store';
import { HudRoot } from './HudRoot';

vi.mock('@/components/OutputCanvas', () => ({ OutputCanvas: () => null }));

describe('startup failure', () => {
  beforeEach(() => {
    useAppStore.setState({ files: [], selectedFileId: null, initialized: false, initError: null });
  });

  it('shows the error instead of a drop zone that can never process anything', () => {
    useAppStore.setState({ initError: 'WebGPU adapter not available' });
    render(<HudRoot />);
    expect(screen.getByRole('alert')).toHaveTextContent('WebGPU adapter not available');
    expect(screen.queryByText('Drop RAW files to start')).not.toBeInTheDocument();
  });

  it('shows the error instead of an endless spinner for a photo added after the failure', () => {
    const file = {
      ...fromLibraryPhoto(fakePhoto()),
      id: 'a',
      status: 'queued',
      result: null,
      originalName: 'a.raf',
      thumbnailUrl: null,
    } as QueuedFile;
    useAppStore.setState({
      initError: "Couldn't download the model list (HTTP 404).",
      files: [file],
      selectedFileId: 'a',
    });
    const { container } = render(<HudRoot />);
    expect(screen.getByRole('alert')).toHaveTextContent('HTTP 404');
    expect(container.querySelector('.xv-stage__spinner')).toBeNull();
  });
});
