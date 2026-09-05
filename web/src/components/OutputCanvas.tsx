import { useEffect, useRef, useMemo, memo } from 'react';
import { useAppStore } from '@/app/store';
import { usePanZoom } from '@/app/hooks/usePanZoom';
import { takeResult } from '@/app/services/processing';
import { createRenderer, isWebGpuSupported, type Renderer } from '@/renderer';
import { configFromPreset, configWithOverrides, computeTonescaleParams } from '@/renderer/grading/opendrt-params';
import type { OpenDrtConfig, PreProcessConfig } from '@/renderer/grading/opendrt-params';
import type { ProcessingResultMeta } from '@/lib/types';

const EMPTY_OVERRIDES: Partial<OpenDrtConfig> = {};
const EMPTY_PREPROCESS: Partial<PreProcessConfig> = {};

interface OutputCanvasProps {
  fileId: string;
  result: ProcessingResultMeta;
}

export const OutputCanvas = memo(function OutputCanvas({ fileId, result }: OutputCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<Renderer | null>(null);
  const rendererKeyRef = useRef('');
  const setRenderer = useAppStore((s) => s.setRenderer);

  // Per-file grading — targeted primitive selectors to avoid re-renders from unrelated file changes
  const lookPreset = useAppStore((s) => s.files.find((f) => f.id === fileId)?.lookPreset ?? 'default');
  const openDrtOverrides = useAppStore((s) => s.files.find((f) => f.id === fileId)?.openDrtOverrides ?? EMPTY_OVERRIDES);
  const preProcessOverrides = useAppStore((s) => s.files.find((f) => f.id === fileId)?.preProcessOverrides ?? EMPTY_PREPROCESS);

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
    if (!canvas || !isWebGpuSupported()) return;

    let cancelled = false;
    const rendererKey = `${fileId}:${hwcW}:${hwcH}:${displayHdr}:${displayHdrHeadroom}`;

    (async () => {
      // Create or reuse renderer
      if (rendererKey !== rendererKeyRef.current) {
        rendererRef.current?.dispose();
        rendererRef.current = null;
        rendererKeyRef.current = '';
        canvas.width = hwcW;
        canvas.height = hwcH;
        const renderer = await createRenderer(canvas,
          displayHdr ? { hdr: true, headroom: displayHdrHeadroom } : undefined,
        );
        if (cancelled) { renderer.dispose(); return; }
        rendererRef.current = renderer;
        rendererKeyRef.current = rendererKey;
      }

      const renderer = rendererRef.current!;
      setRenderer(null);

      if (cancelled) return;
      const image = takeResult(fileId);
      if (!image) {
        // No undisplayed result (e.g. restored session) → re-queue for processing
        useAppStore.getState().updateFileStatus(fileId, 'queued');
        return;
      }

      try {
        renderer.setImage(image.gpu);
      } finally {
        image.dispose();
      }
      const file = useAppStore.getState().files.find((f) => f.id === fileId);
      const preset = file?.lookPreset ?? 'default';
      const overrides = file?.openDrtOverrides ?? {};
      const preProcess = file?.preProcessOverrides ?? {};
      applyOpenDrt(renderer, preset, overrides, preProcess, renderer.display.hdr ? renderer.display.headroom : undefined);
      renderer.render();
      setRenderer(renderer);
    })().catch((error: unknown) => {
      if (cancelled) return;
      const message = error instanceof Error ? error.message : String(error);
      useAppStore.getState().updateFileStatus(fileId, 'error', message);
      console.error('Display failed:', error);
    });

    return () => { cancelled = true; };
  }, [fileId, result, imgW, imgH, hwcW, hwcH, setRenderer, displayHdr, displayHdrHeadroom]);

  // Dispose renderer on unmount only
  useEffect(() => () => {
    rendererRef.current?.dispose();
    rendererRef.current = null;
    rendererKeyRef.current = '';
    useAppStore.getState().setRenderer(null);
  }, []);

  // Re-render when look preset or overrides change (cheap: uniform update + draw)
  useEffect(() => {
    const renderer = rendererRef.current;
    if (!renderer) return;
    applyOpenDrt(renderer, lookPreset, openDrtOverrides, preProcessOverrides, renderer.display.hdr ? renderer.display.headroom : undefined);
    renderer.render();
  }, [lookPreset, openDrtOverrides, preProcessOverrides]);

  return (
    <div
      ref={containerRef}
      className="xv-canvas-host"
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
    </div>
  );
});

function applyOpenDrt(
  renderer: Renderer,
  lookPreset: string,
  overrides: Partial<OpenDrtConfig>,
  preProcess: Partial<PreProcessConfig>,
  hdrHeadroom?: number,
): void {
  const base = configFromPreset(lookPreset as 'base' | 'default', hdrHeadroom);
  const cfg = configWithOverrides(base, overrides, preProcess);
  const ts = computeTonescaleParams(cfg);
  if (hdrHeadroom != null && hdrHeadroom > 1.0) {
    ts.ts_dsc = 1.0;
  }
  renderer.setGrade(cfg, ts);
}
