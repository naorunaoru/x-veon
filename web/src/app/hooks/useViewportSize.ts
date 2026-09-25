import { useEffect, useState, type RefObject } from 'react';

/** Track CSS size and monitor/browser-zoom DPR changes independently. */
export function useViewportSize(ref: RefObject<HTMLElement | null>) {
  const [size, setSize] = useState({ width: 0, height: 0, dpr: window.devicePixelRatio || 1 });
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    let media: MediaQueryList | undefined;
    const measure = () => {
      const { width, height } = element.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      setSize(previous => previous.width === width && previous.height === height && previous.dpr === dpr
        ? previous : { width, height, dpr });
    };
    const watchDpr = () => {
      media?.removeEventListener('change', watchDpr);
      measure();
      media = window.matchMedia(`(resolution: ${window.devicePixelRatio || 1}dppx)`);
      media.addEventListener('change', watchDpr);
    };
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    watchDpr();
    return () => {
      observer.disconnect();
      media?.removeEventListener('change', watchDpr);
    };
  }, [ref]);
  return size;
}
