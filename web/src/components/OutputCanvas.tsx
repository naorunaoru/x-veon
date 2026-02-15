import { useEffect, useRef } from 'react';
import { useAppStore } from '@/store';
import { usePanZoom } from '@/hooks/usePanZoom';
import { toImageDataWithCC } from '@/pipeline/postprocessor';
import { processHdr } from '@/pipeline/hdr-encoder';
import { readHwc } from '@/lib/opfs-storage';
import type { ProcessingResultMeta } from '@/pipeline/types';

interface OutputCanvasProps {
  fileId: string;
  result: ProcessingResultMeta;
}

export function OutputCanvas({ fileId, result }: OutputCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const setCanvasRef = useAppStore((s) => s.setCanvasRef);

  const imgW = result.metadata.width;
  const imgH = result.metadata.height;

  const { transform, isDragging, handlers, scale } = usePanZoom(
    containerRef, imgW, imgH,
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let cancelled = false;
    setCanvasRef(canvas);
    canvas.width = imgW;
    canvas.height = imgH;

    const { width, height, orientation } = result.exportData;

    readHwc(fileId).then((hwc) => {
      if (cancelled || !hwc) return;

      const imageData = result.isHdr
        ? processHdr(hwc, width, height, orientation)
        : toImageDataWithCC(hwc, width, height, orientation);

      const ctx = (result.isHdr
        ? canvas.getContext('2d', { colorSpace: 'rec2100-hlg' as any })
        : canvas.getContext('2d')) as CanvasRenderingContext2D | null;

      if (ctx) {
        ctx.putImageData(imageData, 0, 0);
      }
    });

    return () => {
      cancelled = true;
      setCanvasRef(null);
    };
  }, [fileId, imgW, imgH, setCanvasRef, result]);

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
