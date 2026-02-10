import { useEffect } from 'react';
import { useAppStore } from '@/store';
import { initWasm } from '@/pipeline/raf-decoder';
import { initModel, getBackend } from '@/pipeline/inference';

export function useInit() {
  const setInitialized = useAppStore((s) => s.setInitialized);
  const setInitError = useAppStore((s) => s.setInitError);

  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        await Promise.all([initWasm(), initModel('./model.onnx')]);
        if (cancelled) return;

        const backend = getBackend() ?? 'unknown';

        // Probe HDR support
        const testCanvas = document.createElement('canvas');
        testCanvas.width = 1;
        testCanvas.height = 1;
        const ctx = testCanvas.getContext('2d', {
          colorSpace: 'rec2100-hlg' as any,
        });
        const attrs = (ctx as any)?.getContextAttributes?.();
        const hdr = attrs?.colorSpace === 'rec2100-hlg';

        setInitialized(backend, hdr);
      } catch (e) {
        if (!cancelled) setInitError((e as Error).message);
      }
    }

    init();
    return () => {
      cancelled = true;
    };
  }, [setInitialized, setInitError]);
}
