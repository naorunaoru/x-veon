import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent, within } from '@testing-library/react';
import { Filmstrip } from './Filmstrip';
import { useAppStore } from '@/store';
import type { QueuedFile } from '@/store';

function makeFile(id: string, status: QueuedFile['status'] = 'done'): QueuedFile {
  return {
    id, file: null, name: id, originalName: `${id}.raf`, thumbnailUrl: null,
    metadata: null, cfaType: 'xtrans', status, error: null, progress: null,
    result: null, resultMethod: null, lensProfile: null, lookPreset: 'default',
    openDrtOverrides: {}, preProcessOverrides: {},
  };
}

describe('Filmstrip', () => {
  beforeEach(() => {
    useAppStore.setState({ files: [makeFile('a'), makeFile('b', 'queued')], selectedFileId: 'a' });
  });

  it('renders one cell per file', () => {
    render(<Filmstrip />);
    expect(screen.getAllByTestId('filmstrip-thumb')).toHaveLength(2);
  });

  it('selects a file on click', () => {
    // Stub impl: assert the call, skip the real action's IDB/OPFS side effects
    // (fire-and-forget, .catch-guarded, but noisy in jsdom).
    const spy = vi.spyOn(useAppStore.getState(), 'selectFile').mockImplementation(() => {});
    render(<Filmstrip />);
    fireEvent.click(screen.getAllByTestId('filmstrip-thumb')[1]);
    expect(spy).toHaveBeenCalledWith('b');
  });

  it('removes a file via the hover-× without selecting it', () => {
    const select = vi.spyOn(useAppStore.getState(), 'selectFile').mockImplementation(() => {});
    const remove = vi.spyOn(useAppStore.getState(), 'removeFile').mockImplementation(() => {});
    render(<Filmstrip />);
    const cellB = screen.getAllByTestId('filmstrip-thumb')[1];
    fireEvent.click(within(cellB).getByLabelText('Remove file'));
    expect(remove).toHaveBeenCalledWith('b');
    expect(select).not.toHaveBeenCalled();
  });

  it('marks the selected cell', () => {
    render(<Filmstrip />);
    expect(screen.getAllByTestId('filmstrip-thumb')[0].className).toMatch(/is-selected/);
  });
});
