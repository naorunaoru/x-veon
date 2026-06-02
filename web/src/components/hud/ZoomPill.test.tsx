import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ZoomPill } from './ZoomPill';
import { useAppStore } from '@/store';

describe('ZoomPill', () => {
  beforeEach(() => useAppStore.setState({ viewScale: 0.2, viewFitScale: 0.2, viewControls: null }));

  it('shows Fit when at the fit scale', () => {
    render(<ZoomPill />);
    expect(screen.getByTestId('zoom-readout').textContent).toBe('Fit');
  });

  it('shows a percentage when zoomed in', () => {
    useAppStore.setState({ viewScale: 0.66, viewFitScale: 0.2 });
    render(<ZoomPill />);
    expect(screen.getByTestId('zoom-readout').textContent).toBe('66%');
  });

  it('the Fit button calls resetView', () => {
    const resetView = vi.fn();
    useAppStore.setState({ viewScale: 0.66, viewFitScale: 0.2, viewControls: { zoomTo: vi.fn(), resetView, panTo: vi.fn() } });
    render(<ZoomPill />);
    screen.getByRole('button', { name: 'Fit' }).click();
    expect(resetView).toHaveBeenCalledTimes(1);
  });
});
