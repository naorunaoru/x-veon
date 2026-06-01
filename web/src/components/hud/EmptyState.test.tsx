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
    const spy = vi.spyOn(useAppStore.getState(), 'addFiles');
    render(<EmptyState />);
    const file = new File(['x'], 'shot.raf', { type: 'image/x-fuji-raf' });
    const zone = screen.getByTestId('empty-dropzone');
    fireEvent.drop(zone, { dataTransfer: { files: [file] } });
    expect(spy).toHaveBeenCalledTimes(1);
    expect(spy.mock.calls[0][0][0].name).toBe('shot.raf');
  });
});
