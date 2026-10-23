import { fromLibraryPhoto } from '@/app/store/photo';
import { fakePhoto, defaultEdit } from '@/test/fake-host';
import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { FileMetaPill } from './FileMetaPill';
import { useAppStore } from '@/app/store';
import type { QueuedFile } from '@/app/store';

function makeFile(over: Partial<QueuedFile>): QueuedFile {
  return {
    ...fromLibraryPhoto(fakePhoto()),
    id: 'a',
    name: 'DSCF8037',
    originalName: 'DSCF8037.RAF',
    thumbnailUrl: null,
    metadata: null,
    cfaType: 'xtrans',
    status: 'done',
    error: null,
    progress: null,
    result: null,
    resultMethod: null,
    lensProfile: null,
    ...over,
    edit: { ...defaultEdit(), lookPreset: 'default', openDrtOverrides: {}, preProcessOverrides: {} },
  };
}

describe('FileMetaPill', () => {
  beforeEach(() => useAppStore.setState({ files: [], currentId: null }));

  it('renders nothing when no file is selected', () => {
    const { container } = render(<FileMetaPill />);
    expect(container.firstChild).toBeNull();
  });

  it('shows filename, camera (FUJIFILM stripped), lens and focal/aperture', () => {
    useAppStore.setState({
      files: [
        makeFile({
          metadata: { camera: 'FUJIFILM X-T5', lensModel: 'XF35mmF1.4 R', focalLength: 35, fNumber: 1.4 },
        }),
      ],
      currentId: 'a',
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
      currentId: 'a',
    });
    render(<FileMetaPill />);
    expect(screen.getByText(/DSCF8037\.RAF/)).toBeInTheDocument();
    expect(screen.getByText(/X-T5/)).toBeInTheDocument();
  });

  it('carries the editing note as a glyph with a tooltip', () => {
    useAppStore.setState({ files: [makeFile({ editing: 'view-only', editingNote: 'newer schema' })], currentId: 'a' });
    render(<FileMetaPill />);
    const note = screen.getByRole('img', { name: 'View only: newer schema' });
    expect(note).toHaveAttribute('title', 'View only: newer schema');
  });
  it('labels a session note as session edits', () => {
    useAppStore.setState({ files: [makeFile({ editing: 'session', editingNote: 'disk full' })], currentId: 'a' });
    render(<FileMetaPill />);
    const note = screen.getByRole('img', { name: 'Session edits: disk full' });
    expect(note).toHaveAttribute('title', 'Session edits: disk full');
  });
  it('shows no glyph for a saved photo', () => {
    useAppStore.setState({ files: [makeFile({ editing: 'saved', editingNote: null })], currentId: 'a' });
    render(<FileMetaPill />);
    expect(screen.queryByRole('img')).toBeNull();
  });
});
