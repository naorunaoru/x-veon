import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { DropSurface } from './DropSurface';
import { importFiles } from '@/app/services/library';
vi.mock('@/app/services/library', () => ({ importFiles: vi.fn() }));

describe('DropSurface', () => {
  it('renders the empty-state prompt', () => {
    render(<DropSurface />);
    expect(screen.getByText('Drop RAW files to start')).toBeInTheDocument();
    expect(screen.getByText('browse files')).toBeInTheDocument();
  });

  it('opens the file picker when the empty surface is clicked', () => {
    render(<DropSurface />);
    const zone = screen.getByTestId('empty-dropzone');
    const input = zone.querySelector('input[type="file"]') as HTMLInputElement;
    const clickSpy = vi.spyOn(input, 'click').mockImplementation(() => {});
    fireEvent.click(zone);
    expect(clickSpy).toHaveBeenCalledTimes(1);
  });

  it('adds picked files via the input', () => {
    // addFiles kicks off a browser-only OPFS write (File.arrayBuffer), so stub it.
    const spy = vi.mocked(importFiles);
    render(<DropSurface />);
    const input = screen.getByTestId('empty-dropzone').querySelector('input[type="file"]')!;
    const file = new File(['x'], 'shot.raf', { type: 'image/x-fuji-raf' });
    fireEvent.change(input, { target: { files: [file] } });
    expect(spy).toHaveBeenCalledTimes(1);
    expect(spy.mock.calls[0][0][0].name).toBe('shot.raf');
  });

  it('renders the overlay variant with a pluralized count pill', () => {
    render(<DropSurface overlay active fileCount={28} />);
    expect(screen.getByText('Drop to add to this session')).toBeInTheDocument();
    expect(screen.getByText(/appending to 28 photos already loaded/)).toBeInTheDocument();
  });

  it('uses the singular form for a single loaded photo', () => {
    render(<DropSurface overlay active fileCount={1} />);
    expect(screen.getByText(/appending to 1 photo already loaded/)).toBeInTheDocument();
  });

  it('toggles the active class', () => {
    const { rerender } = render(<DropSurface />);
    expect(screen.getByTestId('empty-dropzone')).not.toHaveClass('is-active');
    rerender(<DropSurface active />);
    expect(screen.getByTestId('empty-dropzone')).toHaveClass('is-active');
  });
});
