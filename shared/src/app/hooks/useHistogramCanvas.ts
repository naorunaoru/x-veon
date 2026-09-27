import { useEffect, type RefObject } from 'react';
import { useAppStore } from '@/app/store';

/**
 * Register a canvas as a histogram viz target on the GPU renderer and populate
 * it once on mount; unregister on unmount. Multiple components can call this
 * (the always-on HUD widget + the open Scopes panel) and each gets its own
 * live histogram. The renderer schedules the shared frame; this hook only handles
 * registration and requests an initial paint.
 */
export function useHistogramCanvas(canvasRef: RefObject<HTMLCanvasElement | null>) {
  const renderer = useAppStore((s) => s.renderer);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!renderer || !canvas) return;
    renderer.histogram.attach(canvas);
    renderer.requestRender();
    return () => { renderer.histogram.detach(canvas); };
  }, [renderer, canvasRef]);
}
