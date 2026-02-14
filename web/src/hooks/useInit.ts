import { useEffect } from 'react';
import { useAppStore } from '@/store';
import { initWasm } from '@/pipeline/raf-decoder';
import { initModels, getBackend, getModelMeta } from '@/pipeline/inference';
import { initDemosaicGpuSafe } from '@/pipeline/demosaic';

export function useInit() {
  const setInitialized = useAppStore((s) => s.setInitialized);
  const setInitError = useAppStore((s) => s.setInitError);

  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        await Promise.all([initWasm(), initModels(), initDemosaicGpuSafe()]);
        if (cancelled) return;

        const backend = getBackend() ?? 'unknown';

        // Probe HDR support — getContext throws on browsers/GPUs without Canvas HDR
        let hdr = false;
        try {
          const testCanvas = document.createElement('canvas');
          testCanvas.width = 1;
          testCanvas.height = 1;
          const ctx = testCanvas.getContext('2d', {
            colorSpace: 'rec2100-hlg' as any,
          });
          const attrs = (ctx as any)?.getContextAttributes?.();
          hdr = attrs?.colorSpace === 'rec2100-hlg';
        } catch {
          // Canvas HDR not supported — fall back to SDR
        }

        setInitialized(backend, hdr, getModelMeta());
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
