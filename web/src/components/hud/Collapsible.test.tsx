import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { Collapsible } from './Collapsible';

describe('Collapsible', () => {
  it('hides its body until the header is clicked', () => {
    render(<Collapsible title="Purity"><p>inner</p></Collapsible>);
    expect(screen.queryByText('inner')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: /Purity/ }));
    expect(screen.getByText('inner')).toBeInTheDocument();
  });
  it('starts open when defaultOpen', () => {
    render(<Collapsible title="Purity" defaultOpen><p>inner</p></Collapsible>);
    expect(screen.getByText('inner')).toBeInTheDocument();
  });
  it('renders an enable toggle that calls onToggle without toggling the section', () => {
    const onToggle = vi.fn();
    render(
      <Collapsible title="Hue Contrast" enabled={false} onToggle={onToggle}>
        <p>inner</p>
      </Collapsible>,
    );
    fireEvent.click(screen.getByRole('switch', { name: 'Hue Contrast' }));
    expect(onToggle).toHaveBeenCalledWith(true);
    expect(screen.queryByText('inner')).toBeNull();
  });
});
