import { useEffect, useRef, useMemo } from 'react';
import { useAppStore } from '@/store';
import { usePanZoom } from '@/hooks/usePanZoom';
import { readHwc } from '@/lib/opfs-storage';
import { HdrRenderer } from '@/gl/renderer';
import { configFromPreset, computeTonescaleParams } from '@/gl/opendrt-params';
import type { ProcessingResultMeta } from '@/pipeline/types';
// Fallback imports for browsers without WebGL2
import { toImageDataWithCC } from '@/pipeline/postprocessor';
import { processHdr } from '@/pipeline/hdr-encoder';

interface OutputCanvasProps {
  fileId: string;
  result: ProcessingResultMeta;
}

export function OutputCanvas({ fileId, result }: OutputCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<HdrRenderer | null>(null);
  const setCanvasRef = useAppStore((s) => s.setCanvasRef);

  const toneMap = useAppStore((s) => s.toneMap);
  const lookPreset = useAppStore((s) => s.lookPreset);
  const displayHdr = useAppStore((s) => s.displayHdr);
  const displayHdrHeadroom = useAppStore((s) => s.displayHdrHeadroom);

  const imgW = result.metadata.width;
  const imgH = result.metadata.height;

  const { transform, isDragging, handlers, scale } = usePanZoom(
    containerRef, imgW, imgH,
  );

  const orientationIndex = useMemo(() => {
    const o = result.exportData.orientation;
    if (o === 'Rotate90') return 1;
    if (o === 'Rotate180') return 2;
    if (o === 'Rotate270') return 3;
    return 0;
  }, [result.exportData.orientation]);

  // Initialize renderer + upload image (once per file)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let cancelled = false;
    setCanvasRef(canvas);
    canvas.width = imgW;
    canvas.height = imgH;

    const { width, height, orientation } = result.exportData;

    if (HdrRenderer.isSupported()) {
      // WebGL2 path
      const renderer = new HdrRenderer(canvas,
        displayHdr ? { hdr: true, headroom: displayHdrHeadroom } : undefined,
      );
      rendererRef.current = renderer;
      renderer.setOrientation(orientationIndex);

      readHwc(fileId).then((hwc) => {
        if (cancelled || !hwc) return;
        renderer.uploadImage(hwc, width, height);
        applyToneMap(renderer, toneMap, lookPreset, renderer.isHdrDisplay ? renderer.hdrHeadroom : undefined);
        renderer.render();
      });
    } else {
      // Fallback: original 2D canvas path
      rendererRef.current = null;
      readHwc(fileId).then((hwc) => {
        if (cancelled || !hwc) return;

        const imageData = result.isHdr
          ? processHdr(hwc, width, height, orientation)
          : toImageDataWithCC(hwc, width, height, orientation);

        const ctx = (result.isHdr
          ? canvas.getContext('2d', { colorSpace: 'rec2100-hlg' as any })
          : canvas.getContext('2d')) as CanvasRenderingContext2D | null;

        if (ctx) ctx.putImageData(imageData, 0, 0);
      });
    }

    return () => {
      cancelled = true;
      rendererRef.current?.dispose();
      rendererRef.current = null;
      setCanvasRef(null);
    };
  }, [fileId, imgW, imgH, setCanvasRef, result, orientationIndex, displayHdr, displayHdrHeadroom]);

  // Re-render when tone map settings change (cheap: uniform update + draw)
  useEffect(() => {
    const renderer = rendererRef.current;
    if (!renderer) return;
    applyToneMap(renderer, toneMap, lookPreset, renderer.isHdrDisplay ? renderer.hdrHeadroom : undefined);
    renderer.render();
  }, [toneMap, lookPreset]);

  return (
    <div
      ref={containerRef}
      className="w-full h-full overflow-hidden"
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
    </div>
  );
}

function applyToneMap(
  renderer: HdrRenderer,
  toneMap: string,
  lookPreset: string,
  hdrHeadroom?: number,
): void {
  if (toneMap === 'opendrt') {
    const cfg = configFromPreset(lookPreset as 'base' | 'default', hdrHeadroom);
    const ts = computeTonescaleParams(cfg);
    if (hdrHeadroom != null && hdrHeadroom > 1.0) {
      // HDR: don't scale down to SDR range — let values > 1.0 pass through as HDR highlights
      ts.ts_dsc = 1.0;
    }
    renderer.setOpenDrtMode(ts, cfg);
  } else {
    renderer.setLegacyMode();
  }
}
