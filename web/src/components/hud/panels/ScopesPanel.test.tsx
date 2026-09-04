import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ScopesPanel } from './ScopesPanel';
import { useAppStore } from '@/store';

describe('ScopesPanel', () => {
  beforeEach(() => useAppStore.setState({
    openPanel: 'scopes', histogramSource: 'display', histogramChannel: 'rgb',
  }));

  it('renders the histogram mode buttons', () => {
    render(<ScopesPanel />);
    ['Display', 'Scene', 'L', 'RGB', 'EV'].forEach((m) =>
      expect(screen.getByRole('button', { name: m })).toBeInTheDocument());
  });

  it('mode buttons drive the shared store state', () => {
    const src = vi.spyOn(useAppStore.getState(), 'setHistogramSource');
    const ch = vi.spyOn(useAppStore.getState(), 'setHistogramChannel');
    render(<ScopesPanel />);
    fireEvent.click(screen.getByRole('button', { name: 'Scene' }));
    fireEvent.click(screen.getByRole('button', { name: 'EV' }));
    expect(src).toHaveBeenCalledWith('scene');
    expect(ch).toHaveBeenCalledWith('ev');
  });
});
