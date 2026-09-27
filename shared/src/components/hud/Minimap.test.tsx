import { describe, it, expect, beforeEach, vi } from 'vitest';
import { fireEvent, render } from '@testing-library/react';
import { Minimap } from './Minimap';
import { useAppStore } from '@/app/store';
import type { QueuedFile } from '@/app/store';
import type { ProcessingResultMeta } from '@/lib/types';

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

function interactiveMinimap() {
  const panTo = vi.fn();
  useAppStore.setState({ viewControls: { panTo, zoomTo: vi.fn(), resetView: vi.fn() } });
  const { container } = render(<Minimap />);
  const map = container.querySelector<HTMLElement>('.xv-minimap')!;
  map.setPointerCapture = vi.fn();
  map.releasePointerCapture = vi.fn();
  map.getBoundingClientRect = () => ({ left: 12, top: 400, width: 200, height: 132.5 }) as DOMRect;
  const pointer = (type: string, x: number, y: number, target: Element = map, button = 0) => {
    fireEvent(target, Object.assign(new MouseEvent(type, {
      bubbles: true, clientX: 12 + x, clientY: 400 + y, button,
    }), { pointerId: 1 }));
  };
  return { map, panTo, pointer };
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

  it('centers a clicked image point using the thumbnail letterbox offset and zoom', () => {
    useAppStore.setState({ viewScale: 2 });
    const { map, panTo, pointer } = interactiveMinimap();
    // The thumbnail is horizontally letterboxed; choose image point (3000, 750).
    const k = 132.5 / 3000;
    const x = (200 - 4000 * k) / 2 + 3000 * k;
    const y = 750 * k;
    const thumbnail = map.querySelector('img')!;
    pointer('pointerdown', x, y, thumbnail);
    pointer('pointerup', x, y, thumbnail);
    expect(panTo).toHaveBeenCalledExactlyOnceWith({ x: -5600, y: -1200 });
    expect(map.setPointerCapture).toHaveBeenCalledWith(1);
    expect(map.releasePointerCapture).toHaveBeenCalledWith(1);
  });

  it('centers clicks inside the viewport rectangle, allowing small pointer jitter', () => {
    const { map, panTo, pointer } = interactiveMinimap();
    const rect = map.querySelector('.xv-minimap__rect')!;
    const k = 132.5 / 3000;
    const x = (200 - 4000 * k) / 2 + 600 * k;
    const y = 450 * k;
    pointer('pointerdown', x - 1, y, rect);
    pointer('pointermove', x, y, rect);
    pointer('pointerup', x, y, rect);
    expect(panTo).toHaveBeenCalledTimes(1);
    expect(panTo.mock.calls[0][0].x).toBeCloseTo(-200);
    expect(panTo.mock.calls[0][0].y).toBeCloseTo(-150);
  });

  it('keeps relative dragging and does not recenter on release', () => {
    const { panTo, pointer } = interactiveMinimap();
    pointer('pointerdown', 40, 25);
    pointer('pointermove', 60, 35);
    pointer('pointerup', 60, 35);
    expect(panTo).toHaveBeenCalledTimes(1);
    expect(panTo.mock.calls[0][0].x).toBeCloseTo(-400 - 20 / (132.5 / 3000));
    expect(panTo.mock.calls[0][0].y).toBeCloseTo(-300 - 10 / (132.5 / 3000));
  });

  it('maps clicks in the letterbox margin to the nearest image edge', () => {
    const { panTo, pointer } = interactiveMinimap();
    pointer('pointerdown', 0, 66.25);
    pointer('pointerup', 0, 66.25);
    expect(panTo).toHaveBeenCalledExactlyOnceWith({ x: 400, y: -1200 });
  });

  it.each(['pointercancel', 'lostpointercapture'])('does not pan after %s', (event) => {
    const { map, panTo, pointer } = interactiveMinimap();
    pointer('pointerdown', 100, 66.25);
    pointer(event, 100, 66.25);
    pointer('pointermove', 120, 66.25);
    pointer('pointerup', 120, 66.25);
    expect(panTo).not.toHaveBeenCalled();
    expect(map).not.toHaveClass('is-dragging');
  });

  it('ignores secondary-button clicks', () => {
    const { map, panTo, pointer } = interactiveMinimap();
    pointer('pointerdown', 100, 66.25, map, 2);
    pointer('pointerup', 100, 66.25, map, 2);
    expect(panTo).not.toHaveBeenCalled();
    expect(map.setPointerCapture).not.toHaveBeenCalled();
  });
});
