// Pointer interaction hooks for the Rendering panel's spatial controls.
//
// Both attach their move/up listeners to `window` while a gesture is active and
// read the live handler through a ref, so the latest closure (current params)
// is always used — the React-state value captured at pointerdown never goes
// stale mid-drag. Pointer events (not mouse events) per the production handoff.
import * as React from 'react';

export const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));
export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

/** Pointer position relative to an element's top-left, plus its size. */
export function relPos(e: PointerEvent | React.PointerEvent, el: Element) {
  const r = el.getBoundingClientRect();
  return { x: e.clientX - r.left, y: e.clientY - r.top, w: r.width, h: r.height };
}

/** Generic pointer-drag on an SVG handle. `onMove` receives every pointermove. */
export function useDrag(onMove: (e: PointerEvent) => void) {
  const ref = React.useRef(onMove);
  ref.current = onMove;
  const [active, setActive] = React.useState(false);

  React.useEffect(() => {
    if (!active) return;
    const mv = (e: PointerEvent) => ref.current(e);
    const up = () => setActive(false);
    window.addEventListener('pointermove', mv);
    window.addEventListener('pointerup', up);
    return () => {
      window.removeEventListener('pointermove', mv);
      window.removeEventListener('pointerup', up);
    };
  }, [active]);

  const start = (e: React.PointerEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setActive(true);
    ref.current(e.nativeEvent);
  };
  return { start, active };
}

interface ScrubConfig {
  getValue: () => number;
  apply: (v: number) => void;
  min: number;
  max: number;
  /** Vertical pixels to traverse the whole [min,max] range. */
  pxFull?: number;
}

/**
 * DAW-style scrubby value: press on a number and drag ↕ to adjust.
 * Up = increase, down = decrease; Shift = fine (0.2×). The whole range is
 * traversed over `pxFull` pixels of vertical travel.
 */
export function useScrub({ getValue, apply, min, max, pxFull = 200 }: ScrubConfig) {
  const ref = React.useRef({ getValue, apply, min, max, pxFull });
  ref.current = { getValue, apply, min, max, pxFull };
  const start = React.useRef<{ y: number; v: number } | null>(null);
  const [active, setActive] = React.useState(false);

  React.useEffect(() => {
    if (!active) return;
    const mv = (e: PointerEvent) => {
      const s = start.current;
      if (!s) return;
      const { apply, min, max, pxFull } = ref.current;
      const range = max - min;
      const mult = e.shiftKey ? 0.2 : 1;
      const dy = s.y - e.clientY; // up positive
      apply(clamp(s.v + (dy / pxFull) * range * mult, min, max));
    };
    const up = () => setActive(false);
    window.addEventListener('pointermove', mv);
    window.addEventListener('pointerup', up);
    const prev = document.body.style.cursor;
    document.body.style.cursor = 'ns-resize';
    return () => {
      window.removeEventListener('pointermove', mv);
      window.removeEventListener('pointerup', up);
      document.body.style.cursor = prev;
    };
  }, [active]);

  const onPointerDown = (e: React.PointerEvent) => {
    e.preventDefault();
    e.stopPropagation();
    start.current = { y: e.clientY, v: ref.current.getValue() };
    setActive(true);
  };
  return { onPointerDown, active };
}
