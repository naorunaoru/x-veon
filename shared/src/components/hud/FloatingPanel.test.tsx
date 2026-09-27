import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { FloatingPanel } from './FloatingPanel';

describe('FloatingPanel', () => {
  it('renders the title and body', () => {
    render(<FloatingPanel title="Exposure" onClose={() => {}}><p>body</p></FloatingPanel>);
    expect(screen.getByText('Exposure')).toBeInTheDocument();
    expect(screen.getByText('body')).toBeInTheDocument();
  });

  it('calls onClose from the close button', () => {
    const onClose = vi.fn();
    render(<FloatingPanel title="Exposure" onClose={onClose}><p>x</p></FloatingPanel>);
    screen.getByLabelText('Close panel').click();
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('shows reset only when modified, and calls onReset', () => {
    const onReset = vi.fn();
    const { rerender } = render(
      <FloatingPanel title="Exposure" onClose={() => {}} onReset={onReset} modified={false}><p>x</p></FloatingPanel>,
    );
    expect(screen.queryByLabelText('Reset to preset')).toBeNull();
    rerender(<FloatingPanel title="Exposure" onClose={() => {}} onReset={onReset} modified><p>x</p></FloatingPanel>);
    screen.getByLabelText('Reset to preset').click();
    expect(onReset).toHaveBeenCalledTimes(1);
  });
});
