import { describe, it, expect, beforeEach } from 'vitest';
import { render } from '@testing-library/react';
import { Minimap } from './Minimap';
import { useAppStore } from '@/store';
import type { QueuedFile } from '@/store';
import type { ProcessingResultMeta } from '@/pipeline/types';

function fileWithResult(): QueuedFile {
  const result = {
    exportData: { width: 4000, height: 3000, xyzToCam: null, wbCoeffs: new Float32Array([1, 1, 1]), camToXyz: new Float32Array(12), orientation: 'Normal' },
    metadata: { make: 'F', model: 'X', width: 4000, height: 3000, tileCount: 1, inferenceTime: 0, backend: 'webgpu', exposureBias: 0, lensModel: '', focalLength: 0, fNumber: 0, colorTemp: 5500, tint: 0 },
  } as unknown as ProcessingResultMeta;
  return {
    id: 'a', file: null, name: 'a', originalName: 'a.raf', thumbnailUrl: 'blob:thumb',
    metadata: null, cfaType: 'xtrans', status: 'done', error: null, progress: null,
    result, resultMethod: 'neural-net', lensProfile: null, lookPreset: 'default',
    openDrtOverrides: {}, preProcessOverrides: {},
  };
}

describe('Minimap', () => {
  beforeEach(() => useAppStore.setState({
    files: [fileWithResult()], selectedFileId: 'a',
    viewScale: 1, viewFitScale: 0.04, viewPan: { x: -400, y: -300 },
    viewContainerW: 800, viewContainerH: 600, viewControls: null,
  }));

  it('renders the viewport rect when zoomed in past fit', () => {
    const { container } = render(<Minimap />);
    expect(container.querySelector('.xv-minimap')).not.toBeNull();
    expect(container.querySelector('.xv-minimap__rect')).not.toBeNull();
  });

  it('renders nothing at fit (not zoomed in)', () => {
    useAppStore.setState({ viewScale: 0.04, viewFitScale: 0.04 });
    const { container } = render(<Minimap />);
    expect(container.firstChild).toBeNull();
  });

  it('renders nothing without a result', () => {
    useAppStore.setState({ files: [], selectedFileId: null });
    const { container } = render(<Minimap />);
    expect(container.firstChild).toBeNull();
  });
});
