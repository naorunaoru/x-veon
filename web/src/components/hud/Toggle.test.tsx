import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Toggle } from './Toggle';

describe('Toggle', () => {
  it('reflects the checked state', () => {
    render(<Toggle checked label="Low contrast" onChange={() => {}} />);
    const btn = screen.getByRole('switch', { name: 'Low contrast' });
    expect(btn).toHaveAttribute('aria-checked', 'true');
    expect(btn.className).toMatch(/is-on/);
  });
  it('calls onChange with the toggled value', () => {
    const onChange = vi.fn();
    render(<Toggle checked={false} label="Low contrast" onChange={onChange} />);
    screen.getByRole('switch', { name: 'Low contrast' }).click();
    expect(onChange).toHaveBeenCalledWith(true);
  });
});
