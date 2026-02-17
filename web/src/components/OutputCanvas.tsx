import { useEffect, useRef, useMemo, useState, memo } from 'react';
import { Loader2 } from 'lucide-react';
import { useAppStore } from '@/store';
import { usePanZoom } from '@/hooks/usePanZoom';
import { readHwc, hwcKey } from '@/lib/opfs-storage';
import { HdrRenderer } from '@/gl/renderer';
import { configFromPreset, configWithOverrides, computeTonescaleParams } from '@/gl/opendrt-params';
import type { OpenDrtConfig } from '@/gl/opendrt-params';
import type { ProcessingResultMeta } from '@/pipeline/types';

const EMPTY_OVERRIDES: Partial<OpenDrtConfig> = {};

interface OutputCanvasProps {
  fileId: string;
  result: ProcessingResultMeta;
}

export const OutputCanvas = memo(function OutputCanvas({ fileId, result }: OutputCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<HdrRenderer | null>(null);
  const rendererKeyRef = useRef('');
  const [loadingHwc, setLoadingHwc] = useState(true);
  const setCanvasRef = useAppStore((s) => s.setCanvasRef);

  // Per-file grading — targeted primitive selectors to avoid re-renders from unrelated file changes
  const lookPreset = useAppStore((s) => s.files.find((f) => f.id === fileId)?.lookPreset ?? 'default');
  const openDrtOverrides = useAppStore((s) => s.files.find((f) => f.id === fileId)?.openDrtOverrides ?? EMPTY_OVERRIDES);

  const displayHdr = useAppStore((s) => s.displayHdr);
  const displayHdrHeadroom = useAppStore((s) => s.displayHdrHeadroom);

  const imgW = result.metadata.width;
  const imgH = result.metadata.height;
  const hwcW = result.exportData.width;
  const hwcH = result.exportData.height;

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

  // Create/reuse renderer + load HWC image.
  // The renderer is only recreated when dimensions, orientation, or HDR mode change.
  // Method changes only reload the HWC texture — no renderer destruction, zoom preserved.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !HdrRenderer.isSupported()) return;

    let cancelled = false;
    const rendererKey = `${fileId}:${imgW}:${imgH}:${orientationIndex}:${displayHdr}:${displayHdrHeadroom}`;

    if (rendererKey !== rendererKeyRef.current) {
      rendererRef.current?.dispose();
      canvas.width = imgW;
      canvas.height = imgH;
      const renderer = new HdrRenderer(canvas,
        displayHdr ? { hdr: true, headroom: displayHdrHeadroom } : undefined,
      );
      rendererRef.current = renderer;
      renderer.setOrientation(orientationIndex);
      rendererKeyRef.current = rendererKey;
    }

    const renderer = rendererRef.current!;
    setCanvasRef(null);
    setLoadingHwc(true);

    const method = useAppStore.getState().files.find((f) => f.id === fileId)?.resultMethod ?? null;
    readHwc(method ? hwcKey(fileId, method) : fileId).then((hwc) => {
      if (cancelled || !hwc) return;
      renderer.uploadImage(hwc, hwcW, hwcH);
      const file = useAppStore.getState().files.find((f) => f.id === fileId);
      const preset = file?.lookPreset ?? 'default';
      const overrides = file?.openDrtOverrides ?? {};
      applyOpenDrt(renderer, preset, overrides, renderer.isHdrDisplay ? renderer.hdrHeadroom : undefined);
      renderer.render();
      setCanvasRef(canvas);
      setLoadingHwc(false);
    });

    return () => { cancelled = true; };
  }, [fileId, result, imgW, imgH, hwcW, hwcH, setCanvasRef, orientationIndex, displayHdr, displayHdrHeadroom]);

  // Dispose renderer on unmount only
  useEffect(() => () => {
    rendererRef.current?.dispose();
    rendererRef.current = null;
    rendererKeyRef.current = '';
    useAppStore.getState().setCanvasRef(null);
  }, []);

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
});

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
