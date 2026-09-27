import { afterEach, expect, it, vi } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { useViewportSize } from './useViewportSize';

afterEach(() => vi.unstubAllGlobals());

it('updates for resize and DPR changes and removes observers on unmount', () => {
  let resize!: () => void;
  const disconnect = vi.fn();
  vi.stubGlobal('ResizeObserver', class {
    constructor(callback: () => void) { resize = callback; }
    observe() {}
    disconnect = disconnect;
  });
  const media: { addEventListener: ReturnType<typeof vi.fn>; removeEventListener: ReturnType<typeof vi.fn> }[] = [];
  vi.stubGlobal('matchMedia', vi.fn(() => {
    const query = { addEventListener: vi.fn(), removeEventListener: vi.fn() };
    media.push(query);
    return query;
  }));
  vi.stubGlobal('devicePixelRatio', 1);
  const element = document.createElement('div');
  let width = 800;
  element.getBoundingClientRect = () => ({ width, height: 600 } as DOMRect);
  const ref = { current: element };
  const { result, unmount } = renderHook(() => useViewportSize(ref));
  expect(result.current).toEqual({ width: 800, height: 600, dpr: 1 });
  act(() => { width = 900; resize(); });
  expect(result.current.width).toBe(900);
  act(() => {
    vi.stubGlobal('devicePixelRatio', 2);
    media[0].addEventListener.mock.calls[0][1]();
  });
  expect(result.current.dpr).toBe(2);
  expect(media[0].removeEventListener).toHaveBeenCalled();
  expect(matchMedia).toHaveBeenLastCalledWith('(resolution: 2dppx)');
  unmount();
  expect(disconnect).toHaveBeenCalledOnce();
  expect(media[1].removeEventListener).toHaveBeenCalledOnce();
});
