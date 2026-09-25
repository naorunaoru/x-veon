import { useEffect, useRef, useMemo, memo } from 'react';
import { useAppStore } from '@/app/store';
import { usePanZoom } from '@/app/hooks/usePanZoom';
import { acquireResult } from '@/app/services/processing';
import { createRenderer, isWebGpuSupported, type Renderer } from '@/renderer';
import { configFromPreset, configWithOverrides, computeTonescaleParams } from '@/renderer/grading/opendrt-params';
import type { OpenDrtConfig, PreProcessConfig } from '@/renderer/grading/opendrt-params';
import type { LookPreset, ProcessingResultMeta } from '@/lib/types';

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
  const leaseRef = useRef<ResultLease | null>(null);
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

  // Create the renderer once per canvas, then show the file's current result in it. The
  // result is borrowed, not taken: it stays cached for re-showing this photo, and a new result
  // for the same file (another method) only swaps the image — zoom and renderer are preserved.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !isWebGpuSupported()) return;

    let cancelled = false;

    (async () => {
      if (!rendererRef.current) {
        const { displayHdr: hdr, displayHdrHeadroom: headroom } = useAppStore.getState();
        const created = await createRenderer(canvas, hdr ? { hdr: true, headroom } : undefined);
        if (cancelled || rendererRef.current) { created.dispose(); return; }
        rendererRef.current = created;
      }
      const renderer = rendererRef.current;
      setRenderer(null);

      if (cancelled) return;
      const lease = acquireResult(fileId);
      if (!lease) {
        // No result in memory (e.g. restored session, or evicted) → re-queue for processing
        useAppStore.getState().updateFileStatus(fileId, 'queued');
        return;
      }

      try {
        if (canvas.width !== hwcW || canvas.height !== hwcH) {
          canvas.width = hwcW;
          canvas.height = hwcH;
        }
        renderer.setImage(lease.image.gpu);
      } catch (error) {
        lease.release();
        throw error;
      }
      // The renderer shows the new image now; give the previous one back.
      leaseRef.current?.release();
      leaseRef.current = lease;

      syncDisplay(renderer);
      applyFileGrade(renderer, fileId);
      renderer.render();
      setRenderer(renderer);
    })().catch((error: unknown) => {
      if (cancelled) return;
      const message = error instanceof Error ? error.message : String(error);
      useAppStore.getState().updateFileStatus(fileId, 'error', message);
      console.error('Display failed:', error);
    });

    return () => { cancelled = true; };
  }, [fileId, result, hwcW, hwcH, setRenderer]);

  // HDR on/off or a new headroom: reconfigure the canvas in place — no reprocessing.
  useEffect(() => {
    const renderer = rendererRef.current;
    if (!renderer || !syncDisplay(renderer)) return;
    applyFileGrade(renderer, fileId);
    renderer.render();
  }, [displayHdr, displayHdrHeadroom, fileId]);

  // Dispose the renderer and give the image back on unmount only
  useEffect(() => () => {
    rendererRef.current?.dispose();
    rendererRef.current = null;
    leaseRef.current?.release();
    leaseRef.current = null;
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

type ResultLease = NonNullable<ReturnType<typeof acquireResult>>;

/** Match the renderer's output to the store's display state; true if it changed. */
function syncDisplay(renderer: Renderer): boolean {
  const { displayHdr, displayHdrHeadroom } = useAppStore.getState();
  const headroom = displayHdr ? displayHdrHeadroom : 1.0;
  if (renderer.display.hdr === displayHdr && renderer.display.headroom === headroom) return false;
  renderer.setDisplay({ hdr: displayHdr, headroom });
  return true;
}

/** The file's own look and overrides, tone-mapped for the renderer's current display. */
function applyFileGrade(renderer: Renderer, fileId: string): void {
  const file = useAppStore.getState().files.find((f) => f.id === fileId);
  applyOpenDrt(
    renderer, file?.lookPreset ?? 'default', file?.openDrtOverrides ?? {}, file?.preProcessOverrides ?? {},
    renderer.display.hdr ? renderer.display.headroom : undefined,
  );
}

function applyOpenDrt(
  renderer: Renderer,
  lookPreset: LookPreset,
  overrides: Partial<OpenDrtConfig>,
  preProcess: Partial<PreProcessConfig>,
  hdrHeadroom?: number,
): void {
  const base = configFromPreset(lookPreset, hdrHeadroom);
  const cfg = configWithOverrides(base, overrides, preProcess);
  const ts = computeTonescaleParams(cfg);
  if (hdrHeadroom != null && hdrHeadroom > 1.0) {
    ts.ts_dsc = 1.0;
  }
  renderer.setGrade(cfg, ts);
}
