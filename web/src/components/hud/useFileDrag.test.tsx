import { describe, it, expect, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useFileDrag } from './useFileDrag';

// jsdom has no DataTransfer, so fake the slice each handler reads.
function fileDrag(type: string, files: File[] = []) {
  const evt = new Event(type, { bubbles: true, cancelable: true });
  Object.defineProperty(evt, 'dataTransfer', { value: { types: ['Files'], files } });
  return evt;
}
function textDrag(type: string) {
  const evt = new Event(type, { bubbles: true, cancelable: true });
  Object.defineProperty(evt, 'dataTransfer', { value: { types: ['text/plain'], files: [] } });
  return evt;
}

describe('useFileDrag', () => {
  it('flips dragging on for a file drag and off on drop', () => {
    const onDrop = vi.fn();
    const { result } = renderHook(() => useFileDrag(onDrop));
    expect(result.current).toBe(false);

    act(() => { window.dispatchEvent(fileDrag('dragenter')); });
    expect(result.current).toBe(true);

    const file = new File(['x'], 'a.raf');
    act(() => { window.dispatchEvent(fileDrag('drop', [file])); });
    expect(result.current).toBe(false);
    expect(onDrop).toHaveBeenCalledTimes(1);
    expect(onDrop.mock.calls[0][0][0].name).toBe('a.raf');
  });

  it('only flips off at depth 0 across nested enter/leave', () => {
    const { result } = renderHook(() => useFileDrag(vi.fn()));
    act(() => { window.dispatchEvent(fileDrag('dragenter')); }); // depth 1
    act(() => { window.dispatchEvent(fileDrag('dragenter')); }); // depth 2 (entered a child)
    act(() => { window.dispatchEvent(fileDrag('dragleave')); }); // depth 1 — still dragging
    expect(result.current).toBe(true);
    act(() => { window.dispatchEvent(fileDrag('dragleave')); }); // depth 0
    expect(result.current).toBe(false);
  });

  it('ignores non-file drags', () => {
    const onDrop = vi.fn();
    const { result } = renderHook(() => useFileDrag(onDrop));
    act(() => { window.dispatchEvent(textDrag('dragenter')); });
    expect(result.current).toBe(false);
    act(() => { window.dispatchEvent(textDrag('drop')); });
    expect(onDrop).not.toHaveBeenCalled();
  });
});
