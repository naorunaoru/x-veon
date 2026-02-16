import { useEffect } from 'react';
import { useAppStore } from '@/store';
import { initWasm } from '@/pipeline/raf-decoder';
import { initModels, getBackend, getModelMeta } from '@/pipeline/inference';
import { initDemosaicGpuSafe } from '@/pipeline/demosaic';
import { clearAllHwc } from '@/lib/opfs-storage';
import { probeHdrDisplay } from '@/gl/hdr-display';

export function useInit() {
  const setInitialized = useAppStore((s) => s.setInitialized);
  const setInitError = useAppStore((s) => s.setInitError);

  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        await Promise.all([initWasm(), initModels(), initDemosaicGpuSafe(), clearAllHwc()]);
        if (cancelled) return;

        const backend = getBackend() ?? 'unknown';

        // Probe 2D canvas HDR (rec2100-hlg for export preview)
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

        // Probe WebGL2 display HDR (float16 backbuffer + extended range)
        const hdrDisplayInfo = probeHdrDisplay();
        if (hdrDisplayInfo.supported) {
          useAppStore.getState().setDisplayHdr(true, hdrDisplayInfo.headroom);
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
