import { describe, it, expect } from 'vitest';
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
});
