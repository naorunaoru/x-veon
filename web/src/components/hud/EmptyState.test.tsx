import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { EmptyState } from './EmptyState';
import { useAppStore } from '@/store';

describe('EmptyState', () => {
  it('renders the prompt', () => {
    render(<EmptyState />);
    expect(screen.getByText('Drop RAW files here')).toBeInTheDocument();
  });

  it('adds dropped files', () => {
    // Stub the implementation: we only assert the call, and the real addFiles
    // kicks off a fire-and-forget OPFS write that calls File.arrayBuffer(),
    // which jsdom's File doesn't implement (browser-only — irrelevant here).
    const spy = vi.spyOn(useAppStore.getState(), 'addFiles').mockImplementation(() => {});
    render(<EmptyState />);
    const file = new File(['x'], 'shot.raf', { type: 'image/x-fuji-raf' });
    const zone = screen.getByTestId('empty-dropzone');
    fireEvent.drop(zone, { dataTransfer: { files: [file] } });
    expect(spy).toHaveBeenCalledTimes(1);
    expect(spy.mock.calls[0][0][0].name).toBe('shot.raf');
  });
});
