import { beforeEach, describe, expect, it } from 'vitest';
import { act, render } from '@testing-library/react';
import { RenderingPanel } from '../RenderingPanel';
import { useAppStore, type QueuedFile } from '@/app/store';
import { configFromPreset, configWithOverrides, OPENDRT_LIMITS } from '@/renderer/grading/opendrt-params';

const file = { id: 'a', file: null, name: 'a', originalName: 'a.raf', thumbnailUrl: null, metadata: null,
  cfaType: 'xtrans', status: 'done', error: null, progress: null, result: null, resultMethod: null,
  lensProfile: null, lookPreset: 'default', openDrtOverrides: {}, preProcessOverrides: {} } as QueuedFile;
// Handles render in this order: toe, grey, contrast, highlights.
const KEYS = ['tn_toe', 'tn_lg', 'tn_con', 'tn_sh'] as const;
const GH = 184 - 8 - 18;
const effective = () => configWithOverrides(configFromPreset('default'), useAppStore.getState().files[0].openDrtOverrides);

function handle(container: HTMLElement, i: number) {
  const g = container.querySelectorAll('svg.xv-rsvg > g[style]')[i] as SVGGElement;
  const dot = g.querySelectorAll('circle')[1];
  return { g, x: Number(dot.getAttribute('cx')), y: Number(dot.getAttribute('cy')) };
}
const pointer = (target: EventTarget, type: string, x: number, y: number) =>
  act(() => { target.dispatchEvent(new MouseEvent(type, { bubbles: true, cancelable: true, clientX: x, clientY: y, button: 0 })); });

describe('tone curve handles', () => {
  beforeEach(() => useAppStore.setState({ files: [file], selectedFileId: 'a', displayHdr: false }));

  it.each(KEYS.map((key, i) => [key, i] as const))('pressing the %s handle without moving leaves it unchanged', (key, i) => {
    const { container } = render(<RenderingPanel />);
    const h = handle(container, i);
    const before = effective()[key];
    pointer(h.g, 'pointerdown', h.x, h.y);
    pointer(window, 'pointerup', h.x, h.y);
    expect(effective()[key]).toBe(before);
  });

  it('moves contrast by the pointer travel, within the render limits', () => {
    const { container } = render(<RenderingPanel />);
    const h = handle(container, 2);
    const [lo, hi] = OPENDRT_LIMITS.tn_con;
    pointer(h.g, 'pointerdown', h.x, h.y);
    pointer(window, 'pointermove', h.x, h.y - GH / 10);  // a tenth of the graph upwards
    expect(effective().tn_con).toBeCloseTo(1.4 + (hi - lo) / 10, 2);
    pointer(window, 'pointermove', h.x, h.y - 10 * GH);
    expect(effective().tn_con).toBe(hi);
    pointer(window, 'pointerup', h.x, h.y);
  });
});
