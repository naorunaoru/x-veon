import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Collapsible } from './Collapsible';

describe('Collapsible', () => {
  it('hides its body until the header is clicked', () => {
    render(<Collapsible title="Purity"><p>inner</p></Collapsible>);
    expect(screen.queryByText('inner')).toBeNull();
    screen.getByRole('button', { name: /Purity/ }).click();
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
    screen.getByRole('switch', { name: 'Hue Contrast' }).click();
    expect(onToggle).toHaveBeenCalledWith(true);
    expect(screen.queryByText('inner')).toBeNull();
  });
});
