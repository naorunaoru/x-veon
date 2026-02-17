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
  const setRendererRef = useAppStore((s) => s.setRendererRef);

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

  // CSS rotation correction: rotate canvas to match EXIF orientation.
  // Canvas stays at unrotated (texture) dimensions; CSS handles visual rotation.
  const rotationCss = useMemo(() => {
    const o = result.exportData.orientation;
    if (o === 'Rotate90') return `translate(${hwcH}px, 0px) rotate(90deg)`;
    if (o === 'Rotate180') return `translate(${hwcW}px, ${hwcH}px) rotate(180deg)`;
    if (o === 'Rotate270') return `translate(0px, ${hwcW}px) rotate(270deg)`;
    return '';
  }, [result.exportData.orientation, hwcW, hwcH]);

  // Create/reuse renderer + load HWC image.
  // The renderer is only recreated when dimensions or HDR mode change.
  // Method changes only reload the HWC texture — no renderer destruction, zoom preserved.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !HdrRenderer.isSupported()) return;

    let cancelled = false;
    const rendererKey = `${fileId}:${hwcW}:${hwcH}:${displayHdr}:${displayHdrHeadroom}`;

    if (rendererKey !== rendererKeyRef.current) {
      rendererRef.current?.dispose();
      canvas.width = hwcW;
      canvas.height = hwcH;
      const renderer = new HdrRenderer(canvas,
        displayHdr ? { hdr: true, headroom: displayHdrHeadroom } : undefined,
      );
      rendererRef.current = renderer;
      rendererKeyRef.current = rendererKey;
    }

    const renderer = rendererRef.current!;
    setCanvasRef(null);
    setRendererRef(null);
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
      setRendererRef(renderer);
      setLoadingHwc(false);
    });

    return () => { cancelled = true; };
  }, [fileId, result, imgW, imgH, hwcW, hwcH, setCanvasRef, setRendererRef, displayHdr, displayHdrHeadroom]);

  // Dispose renderer on unmount only
  useEffect(() => () => {
    rendererRef.current?.dispose();
    rendererRef.current = null;
    rendererKeyRef.current = '';
    useAppStore.getState().setCanvasRef(null);
    useAppStore.getState().setRendererRef(null);
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
          transform: rotationCss ? `${transform} ${rotationCss}` : transform,
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
