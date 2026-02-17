import { useEffect, useRef, useState } from 'react';
import { cn } from '@/lib/utils';
import { useAppStore } from '@/store';

type Mode = 'luma' | 'rgb';

const BINS = 256;
const SAMPLE_W = 320;
const SAMPLE_H = 320;

interface HistogramData {
  r: Uint32Array;
  g: Uint32Array;
  b: Uint32Array;
  l: Uint32Array;
}

// Offscreen canvas reused across calls
let _sampleCanvas: OffscreenCanvas | null = null;
let _sampleCtx: OffscreenCanvasRenderingContext2D | null = null;

function getSampleCtx(): OffscreenCanvasRenderingContext2D | null {
  if (!_sampleCtx) {
    _sampleCanvas = new OffscreenCanvas(SAMPLE_W, SAMPLE_H);
    _sampleCtx = _sampleCanvas.getContext('2d', { willReadFrequently: true });
  }
  return _sampleCtx;
}

function computeHistogram(source: HTMLCanvasElement): HistogramData | null {
  const ctx = getSampleCtx();
  if (!ctx || source.width === 0 || source.height === 0) return null;

  ctx.clearRect(0, 0, SAMPLE_W, SAMPLE_H);
  ctx.drawImage(source, 0, 0, SAMPLE_W, SAMPLE_H);
  const { data } = ctx.getImageData(0, 0, SAMPLE_W, SAMPLE_H);

  const r = new Uint32Array(BINS);
  const g = new Uint32Array(BINS);
  const b = new Uint32Array(BINS);
  const l = new Uint32Array(BINS);

  for (let i = 0; i < data.length; i += 4) {
    const ri = data[i], gi = data[i + 1], bi = data[i + 2];
    r[ri]++;
    g[gi]++;
    b[bi]++;
    // Rec.709 luminance
    const lum = Math.min(255, Math.round(0.2126 * ri + 0.7152 * gi + 0.0722 * bi));
    l[lum]++;
  }

  return { r, g, b, l };
}

function drawChannel(
  ctx: CanvasRenderingContext2D,
  bins: Uint32Array,
  w: number,
  h: number,
  maxLog: number,
  fillColor: string,
  strokeColor: string,
) {
  ctx.beginPath();
  ctx.moveTo(0, h);
  for (let i = 0; i < BINS; i++) {
    const x = (i / (BINS - 1)) * w;
    const y = bins[i] > 0 ? h - (Math.log(1 + bins[i]) / maxLog) * h : h;
    ctx.lineTo(x, y);
  }
  ctx.lineTo(w, h);
  ctx.closePath();
  ctx.fillStyle = fillColor;
  ctx.fill();
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 1;
  ctx.stroke();
}

function drawHistogram(
  canvas: HTMLCanvasElement,
  data: HistogramData,
  mode: Mode,
): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  // Find global max for consistent Y scaling
  let maxCount = 0;
  const channels = mode === 'luma' ? [data.l] : [data.r, data.g, data.b];
  for (const ch of channels) {
    // Skip first and last bins (pure black/white) for better scaling
    for (let i = 1; i < BINS - 1; i++) {
      if (ch[i] > maxCount) maxCount = ch[i];
    }
  }
  if (maxCount === 0) return;
  const maxLog = Math.log(1 + maxCount);

  if (mode === 'luma') {
    drawChannel(ctx, data.l, w, h, maxLog, 'rgba(200, 200, 200, 0.7)', 'rgba(255, 255, 255, 0.4)');
  } else {
    ctx.globalCompositeOperation = 'screen';
    drawChannel(ctx, data.r, w, h, maxLog, 'rgba(220, 50, 50, 0.6)', 'rgba(255, 80, 80, 0.5)');
    drawChannel(ctx, data.g, w, h, maxLog, 'rgba(50, 200, 50, 0.6)', 'rgba(80, 255, 80, 0.5)');
    drawChannel(ctx, data.b, w, h, maxLog, 'rgba(50, 80, 220, 0.6)', 'rgba(80, 120, 255, 0.5)');
    ctx.globalCompositeOperation = 'source-over';
  }
}

export function Histogram() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [mode, setMode] = useState<Mode>('luma');

  const sourceCanvas = useAppStore((s) => s.canvasRef);
  const selectedFile = useAppStore((s) => s.files.find((f) => f.id === s.selectedFileId));
  const lookPreset = selectedFile?.lookPreset ?? 'default';
  const overrides = selectedFile?.openDrtOverrides ?? {};

  useEffect(() => {
    if (!sourceCanvas || !canvasRef.current) return;

    const raf = requestAnimationFrame(() => {
      const data = computeHistogram(sourceCanvas);
      if (data && canvasRef.current) {
        drawHistogram(canvasRef.current, data, mode);
      }
    });

    return () => cancelAnimationFrame(raf);
  }, [sourceCanvas, lookPreset, overrides, mode]);

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="text-[11px] uppercase tracking-wider text-muted-foreground">Histogram</span>
        <div className="flex rounded bg-muted p-0.5 gap-0.5">
          {(['luma', 'rgb'] as const).map((m) => (
            <button
              key={m}
              className={cn(
                'px-1.5 py-0 text-[10px] font-medium rounded-sm transition-colors',
                mode === m
                  ? 'bg-background text-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground',
              )}
              onClick={() => setMode(m)}
            >
              {m === 'luma' ? 'L' : 'RGB'}
            </button>
          ))}
        </div>
      </div>
      <canvas
        ref={canvasRef}
        width={464}
        height={140}
        className="w-full rounded-sm border border-white/15"
        style={{ background: 'rgba(0,0,0,0.3)', height: 70 }}
      />
    </div>
  );
}
