import { useEffect, useRef } from 'react';
import { useAppStore } from '@/store';
import { usePanZoom } from '@/hooks/usePanZoom';
import type { ProcessingResult } from '@/pipeline/types';

interface OutputCanvasProps {
  result: ProcessingResult;
}

export function OutputCanvas({ result }: OutputCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const setCanvasRef = useAppStore((s) => s.setCanvasRef);

  const { transform, isDragging, handlers, scale } = usePanZoom(
    containerRef,
    result.imageData.width,
    result.imageData.height,
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    setCanvasRef(canvas);

    canvas.width = result.imageData.width;
    canvas.height = result.imageData.height;

    // PQ with explicit OOTF avoids browser's variable HLG system gamma
    const ctx = (result.isHdr
      ? canvas.getContext('2d', { colorSpace: 'rec2100-pq' as any })
      : canvas.getContext('2d')) as CanvasRenderingContext2D | null;

    if (ctx) {
      ctx.putImageData(result.imageData, 0, 0);
    }

    return () => {
      setCanvasRef(null);
    };
  }, [result, setCanvasRef]);

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
