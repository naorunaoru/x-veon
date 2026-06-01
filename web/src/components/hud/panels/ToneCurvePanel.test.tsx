import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ToneCurvePanel } from './ToneCurvePanel';
import { useAppStore } from '@/store';
import type { QueuedFile } from '@/store';

function makeFile(): QueuedFile {
  return {
    id: 'a', file: null, name: 'a', originalName: 'a.raf', thumbnailUrl: null,
    metadata: null, cfaType: 'xtrans', status: 'done', error: null, progress: null,
    result: null, resultMethod: null, lensProfile: null, lookPreset: 'default',
    openDrtOverrides: {}, preProcessOverrides: {},
  };
}

describe('ToneCurvePanel', () => {
  beforeEach(() => useAppStore.setState({ openPanel: 'toneCurve', files: [makeFile()], selectedFileId: 'a' }));

  it('renders the curve, the Toe/Shoulder sliders, and a presets dropdown', () => {
    render(<ToneCurvePanel />);
    expect(screen.getByTestId('tone-curve-path')).toBeInTheDocument();
    expect(screen.getByText('Toe')).toBeInTheDocument();
    expect(screen.getByText('Shoulder')).toBeInTheDocument();
    expect(screen.getByLabelText('Tonescale preset')).toBeInTheDocument();
  });

  it('applying a tonescale preset writes overrides', () => {
    const spy = vi.spyOn(useAppStore.getState(), 'setFileOpenDrtOverride');
    render(<ToneCurvePanel />);
    fireEvent.change(screen.getByLabelText('Tonescale preset'), { target: { value: 'umbra' } });
    expect(spy).toHaveBeenCalledWith('a', 'tn_con', expect.any(Number));
  });
});
