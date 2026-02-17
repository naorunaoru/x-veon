import { useCallback, useEffect, useRef, useState, type RefObject } from 'react';

const MAX_SCALE = 32;
const ZOOM_SENSITIVITY = 0.01;

interface PanZoomState {
  scale: number;
  offsetX: number;
  offsetY: number;
}

function computeFitScale(
  containerW: number, containerH: number,
  contentW: number, contentH: number,
): number {
  if (contentW === 0 || contentH === 0) return 1;
  return Math.min(containerW / contentW, containerH / contentH);
}

function centerOffset(
  containerW: number, containerH: number,
  contentW: number, contentH: number,
  scale: number,
): { x: number; y: number } {
  return {
    x: (containerW - contentW * scale) / 2,
    y: (containerH - contentH * scale) / 2,
  };
}

export function usePanZoom(
  containerRef: RefObject<HTMLDivElement | null>,
  contentWidth: number,
  contentHeight: number,
) {
  const [state, setState] = useState<PanZoomState>({
    scale: 1,
    offsetX: 0,
    offsetY: 0,
  });
  const [isDragging, setIsDragging] = useState(false);
  const fitScaleRef = useRef(1);
  const dragStartRef = useRef({ x: 0, y: 0, ox: 0, oy: 0 });

  // Recompute fit scale on container resize
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const update = () => {
      const rect = container.getBoundingClientRect();
      const fs = computeFitScale(rect.width, rect.height, contentWidth, contentHeight);
      fitScaleRef.current = fs;
      const c = centerOffset(rect.width, rect.height, contentWidth, contentHeight, fs);
      setState((prev) => {
        if (prev.scale === fs && prev.offsetX === c.x && prev.offsetY === c.y) return prev;
        return { scale: fs, offsetX: c.x, offsetY: c.y };
      });
    };

    update();

    const ro = new ResizeObserver(update);
    ro.observe(container);
    return () => ro.disconnect();
  }, [containerRef, contentWidth, contentHeight]);

  // Native wheel listener with { passive: false } to allow preventDefault.
  // React's onWheel is passive and cannot prevent browser pinch-zoom.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();

      const rect = container.getBoundingClientRect();
      const cx = e.clientX - rect.left;
      const cy = e.clientY - rect.top;

      if (e.ctrlKey || e.metaKey) {
        // Pinch zoom (trackpad sends ctrlKey), or ctrl/cmd+wheel
        setState((prev) => {
          const factor = Math.exp(-e.deltaY * ZOOM_SENSITIVITY);
          const newScale = Math.max(fitScaleRef.current, Math.min(MAX_SCALE, prev.scale * factor));
          const ratio = newScale / prev.scale;
          return {
            scale: newScale,
            offsetX: cx - (cx - prev.offsetX) * ratio,
            offsetY: cy - (cy - prev.offsetY) * ratio,
          };
        });
      } else {
        // Everything else: two-finger scroll, mouse wheel, shift+wheel → pan
        setState((prev) => ({
          ...prev,
          offsetX: prev.offsetX - e.deltaX,
          offsetY: prev.offsetY - e.deltaY,
        }));
      }
    };

    container.addEventListener('wheel', handleWheel, { passive: false });
    return () => container.removeEventListener('wheel', handleWheel);
  }, [containerRef]);

  const resetView = useCallback(() => {
    const container = containerRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    const fs = computeFitScale(rect.width, rect.height, contentWidth, contentHeight);
    fitScaleRef.current = fs;
    const c = centerOffset(rect.width, rect.height, contentWidth, contentHeight, fs);
    setState({ scale: fs, offsetX: c.x, offsetY: c.y });
  }, [containerRef, contentWidth, contentHeight]);

  const onPointerDown = useCallback((e: React.PointerEvent) => {
    setIsDragging(true);
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    dragStartRef.current = {
      x: e.clientX,
      y: e.clientY,
      ox: state.offsetX,
      oy: state.offsetY,
    };
  }, [state.offsetX, state.offsetY]);

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    if (!isDragging) return;
    const dx = e.clientX - dragStartRef.current.x;
    const dy = e.clientY - dragStartRef.current.y;
    setState((prev) => ({
      ...prev,
      offsetX: dragStartRef.current.ox + dx,
      offsetY: dragStartRef.current.oy + dy,
    }));
  }, [isDragging]);

  const onPointerUp = useCallback((e: React.PointerEvent) => {
    setIsDragging(false);
    (e.target as HTMLElement).releasePointerCapture(e.pointerId);
  }, []);

  const onDoubleClick = useCallback(() => {
    resetView();
  }, [resetView]);

  const transform = `translate(${state.offsetX}px, ${state.offsetY}px) scale(${state.scale})`;

  return {
    transform,
    isDragging,
    handlers: {
      onPointerDown,
      onPointerMove,
      onPointerUp,
      onDoubleClick,
    },
    resetView,
    scale: state.scale,
  };
}
