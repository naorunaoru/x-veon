import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ScopesPanel } from './ScopesPanel';
import { useAppStore } from '@/store';

describe('ScopesPanel', () => {
  beforeEach(() => useAppStore.setState({ openPanel: 'scopes', showClipMask: false, rendererRef: null }));

  it('renders the histogram mode buttons', () => {
    render(<ScopesPanel />);
    ['Display', 'Scene', 'L', 'RGB', 'EV'].forEach((m) =>
      expect(screen.getByRole('button', { name: m })).toBeInTheDocument());
  });

  it('toggles the highlight-clip overlay', () => {
    const spy = vi.spyOn(useAppStore.getState(), 'setShowClipMask');
    render(<ScopesPanel />);
    screen.getByRole('button', { name: /clipping/i }).click();
    expect(spy).toHaveBeenCalledWith(true);
  });

  it('reflects the clip overlay on-state', () => {
    useAppStore.setState({ showClipMask: true });
    const { container } = render(<ScopesPanel />);
    expect(container.querySelector('.xv-toggle.is-on')).not.toBeNull();
  });
});
