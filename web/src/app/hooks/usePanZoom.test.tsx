import { act, renderHook } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { usePanZoom } from './usePanZoom';
import { useAppStore } from '@/app/store';

let resized: () => void;
beforeEach(() => {
  vi.stubGlobal('ResizeObserver', class { constructor(callback: () => void) { resized = callback; } observe() {} disconnect() {} });
});

it.each([[4000, 6000], [6000, 4000], [4000, 4000]])('allows manual zoom to 75%% of Fit for %s × %s', (width, height) => {
  const container = document.createElement('div');
  container.getBoundingClientRect = () => ({ width: 1200, height: 800, left: 0, top: 0 }) as DOMRect;
  const ref = { current: container };
  const fit = Math.min(1200 / width, 800 / height);
  const { result } = renderHook(() => usePanZoom(ref, width, height));
  expect(result.current.scale).toBe(fit);
  act(() => useAppStore.getState().viewControls!.zoomTo(0));
  expect(result.current.scale).toBe(fit * 0.75);
  const transform = result.current.transform;
  act(() => { useAppStore.getState().setOpenPanel('exposure'); resized(); });
  expect(result.current.transform).toBe(transform);
  act(() => result.current.resetView());
  expect(result.current.scale).toBe(fit);
  act(() => container.dispatchEvent(new WheelEvent('wheel', { ctrlKey: true, deltaY: 1000, clientX: 600, clientY: 400 })));
  expect(result.current.scale).toBe(fit * 0.75);
});
