import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Slider } from './Slider';

describe('Slider', () => {
  it('renders the label and the absolute value', () => {
    render(<Slider label="Contrast" value={1.4} defaultValue={1.4} min={0.5} max={2.5} step={0.01} onChange={() => {}} />);
    expect(screen.getByText('Contrast')).toBeInTheDocument();
    expect(screen.getByText('1.40')).toBeInTheDocument();
  });

  it('marks itself modified and shows a signed delta when off-default', () => {
    const { container } = render(
      <Slider label="Contrast" value={1.0} defaultValue={1.4} min={0.5} max={2.5} step={0.01} onChange={() => {}} />,
    );
    expect(container.querySelector('.xv-slider.is-modified')).not.toBeNull();
    expect(screen.getByText('-0.40')).toBeInTheDocument();
  });

  it('is not modified at the default and shows no delta', () => {
    const { container } = render(
      <Slider label="Toe" value={0.003} defaultValue={0.003} min={0} max={0.02} step={0.001} onChange={() => {}} />,
    );
    expect(container.querySelector('.xv-slider.is-modified')).toBeNull();
    expect(screen.getByText('0.003')).toBeInTheDocument();
    expect(container.querySelector('.xv-slider__delta')).toBeNull();
  });

  it('positions the default tick at the default fraction of the range', () => {
    const { container } = render(
      <Slider label="Exposure" value={0} defaultValue={0} min={-5} max={5} step={0.01} onChange={() => {}} />,
    );
    const tick = container.querySelector('.xv-slider__tick') as HTMLElement;
    expect(tick).not.toBeNull();
    expect(tick.style.left).toBe('50%'); // (0 - -5) / (5 - -5) = 0.5
  });

  it('forwards an aria-label from the label', () => {
    render(<Slider label="Shoulder" value={0.5} defaultValue={0.5} min={0} max={1} step={0.01} onChange={() => {}} />);
    expect(screen.getByRole('slider')).toHaveAttribute('aria-label', 'Shoulder');
  });
});
