import { useEffect, type RefObject } from 'react';
import { useAppStore } from '@/app/store';

/**
 * Register a canvas as a histogram viz target on the GPU renderer and populate
 * it once on mount; unregister on unmount. Multiple components can call this
 * (the always-on HUD widget + the open Scopes panel) and each gets its own
 * live histogram. Mode/grading-driven re-renders are owned by the always-mounted
 * HistogramHud, so this hook only handles registration + the initial paint.
 */
export function useHistogramCanvas(canvasRef: RefObject<HTMLCanvasElement | null>) {
  const renderer = useAppStore((s) => s.rendererRef);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!renderer || !canvas) return;
    renderer.addHistogramCanvas(canvas);
    renderer.render(); // paint this target with the current bins immediately
    return () => { renderer.removeHistogramCanvas(canvas); };
  }, [renderer, canvasRef]);
}
