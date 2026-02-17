import { useEffect, useRef, useMemo, useState } from 'react';
import { Loader2 } from 'lucide-react';
import { useAppStore } from '@/store';
import { usePanZoom } from '@/hooks/usePanZoom';
import { readHwc, hwcKey } from '@/lib/opfs-storage';
import { HdrRenderer } from '@/gl/renderer';
import { configFromPreset, configWithOverrides, computeTonescaleParams } from '@/gl/opendrt-params';
import type { OpenDrtConfig } from '@/gl/opendrt-params';
import type { ProcessingResultMeta } from '@/pipeline/types';

interface OutputCanvasProps {
  fileId: string;
  result: ProcessingResultMeta;
}

export function OutputCanvas({ fileId, result }: OutputCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<HdrRenderer | null>(null);
  const [loadingHwc, setLoadingHwc] = useState(true);
  const setCanvasRef = useAppStore((s) => s.setCanvasRef);

  // Per-file grading
  const selectedFile = useAppStore((s) => s.files.find((f) => f.id === fileId));
  const lookPreset = selectedFile?.lookPreset ?? 'default';
  const openDrtOverrides = selectedFile?.openDrtOverrides ?? {};

  const displayHdr = useAppStore((s) => s.displayHdr);
  const displayHdrHeadroom = useAppStore((s) => s.displayHdrHeadroom);

  const imgW = result.metadata.width;
  const imgH = result.metadata.height;

  const { transform, isDragging, handlers, scale } = usePanZoom(
    containerRef, imgW, imgH,
  );

  const orientationIndex = useMemo(() => {
    const o = result.exportData.orientation;
    if (o === 'Rotate90') return 3;
    if (o === 'Rotate180') return 2;
    if (o === 'Rotate270') return 1;
    return 0;
  }, [result.exportData.orientation]);

  // Initialize renderer + upload image (once per file)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    if (!HdrRenderer.isSupported()) {
      console.error('WebGL2 is required but not available');
      return;
    }

    let cancelled = false;
    setCanvasRef(canvas);
    setLoadingHwc(true);
    canvas.width = imgW;
    canvas.height = imgH;

    const { width, height } = result.exportData;

    const renderer = new HdrRenderer(canvas,
      displayHdr ? { hdr: true, headroom: displayHdrHeadroom } : undefined,
    );
    rendererRef.current = renderer;
    renderer.setOrientation(orientationIndex);

    const method = useAppStore.getState().files.find((f) => f.id === fileId)?.resultMethod;
    readHwc(method ? hwcKey(fileId, method) : fileId).then((hwc) => {
      if (cancelled || !hwc) return;
      renderer.uploadImage(hwc, width, height);
      const file = useAppStore.getState().files.find((f) => f.id === fileId);
      const preset = file?.lookPreset ?? 'default';
      const overrides = file?.openDrtOverrides ?? {};
      applyOpenDrt(renderer, preset, overrides, renderer.isHdrDisplay ? renderer.hdrHeadroom : undefined);
      renderer.render();
      setLoadingHwc(false);
    });

    return () => {
      cancelled = true;
      rendererRef.current?.dispose();
      rendererRef.current = null;
      setCanvasRef(null);
    };
  }, [fileId, imgW, imgH, setCanvasRef, result, orientationIndex, displayHdr, displayHdrHeadroom]);

  // Re-render when look preset or overrides change (cheap: uniform update + draw)
  useEffect(() => {
    const renderer = rendererRef.current;
    if (!renderer) return;
    applyOpenDrt(renderer, lookPreset, openDrtOverrides, renderer.isHdrDisplay ? renderer.hdrHeadroom : undefined);
    renderer.render();
  }, [lookPreset, openDrtOverrides]);

  return (
    <div
      ref={containerRef}
      className="w-full h-full overflow-hidden relative"
      style={{ cursor: isDragging ? 'grabbing' : 'grab', touchAction: 'none' }}
      {...handlers}
    >
      <canvas
        ref={canvasRef}
        style={{
          transformOrigin: '0 0',
          transform,
          imageRendering: scale > 1 ? 'pixelated' : 'auto',
        }}
      />
      {loadingHwc && (
        <div className="absolute inset-0 flex items-center justify-center bg-background/50">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </div>
      )}
    </div>
  );
}

function applyOpenDrt(
  renderer: HdrRenderer,
  lookPreset: string,
  overrides: Partial<OpenDrtConfig>,
  hdrHeadroom?: number,
): void {
  const base = configFromPreset(lookPreset as 'base' | 'default', hdrHeadroom);
  const cfg = configWithOverrides(base, overrides);
  const ts = computeTonescaleParams(cfg);
  if (hdrHeadroom != null && hdrHeadroom > 1.0) {
    ts.ts_dsc = 1.0;
  }
  renderer.setOpenDrtMode(ts, cfg);
}
