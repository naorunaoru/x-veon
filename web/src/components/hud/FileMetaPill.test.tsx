import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { FileMetaPill } from './FileMetaPill';
import { useAppStore } from '@/store';
import type { QueuedFile } from '@/store';

function makeFile(over: Partial<QueuedFile>): QueuedFile {
  return {
    id: 'a', file: null, name: 'DSCF8037', originalName: 'DSCF8037.RAF', thumbnailUrl: null,
    metadata: null, cfaType: 'xtrans', status: 'done', error: null, progress: null,
    result: null, resultMethod: null, lensProfile: null, lookPreset: 'default',
    openDrtOverrides: {}, preProcessOverrides: {}, ...over,
  };
}

describe('FileMetaPill', () => {
  beforeEach(() => useAppStore.setState({ files: [], selectedFileId: null }));

  it('renders nothing when no file is selected', () => {
    const { container } = render(<FileMetaPill />);
    expect(container.firstChild).toBeNull();
  });

  it('shows filename, camera (FUJIFILM stripped), lens and focal/aperture', () => {
    useAppStore.setState({
      files: [makeFile({
        metadata: { camera: 'FUJIFILM X-T5', lensModel: 'XF35mmF1.4 R', focalLength: 35, fNumber: 1.4 },
      })],
      selectedFileId: 'a',
    });
    render(<FileMetaPill />);
    expect(screen.getByText(/DSCF8037\.RAF/)).toBeInTheDocument();
    expect(screen.getByText(/X-T5/)).toBeInTheDocument();
    expect(screen.queryByText(/FUJIFILM/)).toBeNull();
    expect(screen.getByText(/XF35mmF1\.4 R/)).toBeInTheDocument();
    expect(screen.getByText(/35mm/)).toBeInTheDocument();
    expect(screen.getByText(/f\/1\.4/)).toBeInTheDocument();
  });

  it('shows just filename + camera before the decode fills lens info', () => {
    useAppStore.setState({
      files: [makeFile({ metadata: { camera: 'FUJIFILM X-T5', lensModel: '', focalLength: 0, fNumber: 0 } })],
      selectedFileId: 'a',
    });
    render(<FileMetaPill />);
    expect(screen.getByText(/DSCF8037\.RAF/)).toBeInTheDocument();
    expect(screen.getByText(/X-T5/)).toBeInTheDocument();
  });
});
