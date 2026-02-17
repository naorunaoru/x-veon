import { useEffect, useRef, useState } from 'react';
import { cn } from '@/lib/utils';
import { useAppStore } from '@/store';
import type { HistogramData } from '@/gl/renderer';

type Mode = 'luma' | 'rgb' | 'ev';

const BINS = 256;
const EV_MIN = -8;
const EV_RANGE = 16;
const ZONE_COUNT = 16;
const ZONE_BAR_H = 16; // logical canvas pixels

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

function drawEvTicks(ctx: CanvasRenderingContext2D, w: number, h: number): void {
  const ticks = [-6, -3, 0, 3, 6];
  ctx.lineWidth = 1;
  ctx.font = '16px sans-serif';
  ctx.textAlign = 'center';

  for (const ev of ticks) {
    const x = ((ev - EV_MIN) / EV_RANGE) * w;
    // Vertical guide line
    ctx.strokeStyle = ev === 0 ? 'rgba(255, 255, 255, 0.35)' : 'rgba(255, 255, 255, 0.15)';
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
    // Label
    ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
    const label = ev === 0 ? '0' : (ev > 0 ? `+${ev}` : `${ev}`);
    ctx.fillText(label, x, h - 4);
  }
}

function drawZoneBar(
  ctx: CanvasRenderingContext2D,
  bins: Uint32Array,
  w: number,
  y: number,
): void {
  const zoneCounts = new Float64Array(ZONE_COUNT);
  let maxZone = 0;
  for (let z = 0; z < ZONE_COUNT; z++) {
    let sum = 0;
    for (let i = 0; i < 16; i++) {
      sum += bins[z * 16 + i];
    }
    zoneCounts[z] = sum;
    if (sum > maxZone) maxZone = sum;
  }
  if (maxZone === 0) return;

  const zoneW = w / ZONE_COUNT;
  for (let z = 0; z < ZONE_COUNT; z++) {
    const t = z / (ZONE_COUNT - 1);
    const brightness = Math.round(30 + t * 200);
    const alpha = 0.15 + 0.85 * (zoneCounts[z] / maxZone);
    ctx.fillStyle = `rgba(${brightness}, ${brightness}, ${brightness}, ${alpha})`;
    ctx.fillRect(z * zoneW, y, zoneW - 1, ZONE_BAR_H);
  }
}

function drawHistogram(
  canvas: HTMLCanvasElement,
  data: HistogramData,
  mode: Mode,
): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const w = canvas.width;
  const fullH = canvas.height;
  const h = mode === 'ev' ? fullH - ZONE_BAR_H - 4 : fullH;
  ctx.clearRect(0, 0, w, fullH);

  // Find global max for consistent Y scaling
  let maxCount = 0;
  const channels = mode === 'rgb' ? [data.r, data.g, data.b] : [data.l];
  for (const ch of channels) {
    // Skip first and last bins (pure black/white) for better scaling
    for (let i = 1; i < BINS - 1; i++) {
      if (ch[i] > maxCount) maxCount = ch[i];
    }
  }
  if (maxCount === 0) return;
  const maxLog = Math.log(1 + maxCount);

  if (mode === 'ev') {
    drawChannel(ctx, data.l, w, h, maxLog, 'rgba(120, 180, 255, 0.6)', 'rgba(160, 210, 255, 0.5)');
    drawEvTicks(ctx, w, h);
    drawZoneBar(ctx, data.l, w, h + 4);
  } else if (mode === 'luma') {
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

  const renderer = useAppStore((s) => s.rendererRef);
  const selectedFile = useAppStore((s) => s.files.find((f) => f.id === s.selectedFileId));
  const lookPreset = selectedFile?.lookPreset ?? 'default';
  const overrides = selectedFile?.openDrtOverrides ?? {};

  useEffect(() => {
    if (!renderer || !canvasRef.current) return;

    renderer.histogramMode = mode === 'ev' ? 'hdr' : 'sdr';

    let cancelled = false;

    renderer.render();
    renderer.getHistogramData().then((data) => {
      if (!cancelled && data && canvasRef.current) {
        drawHistogram(canvasRef.current, data, mode);
      }
    });

    return () => { cancelled = true; };
  }, [renderer, lookPreset, overrides, mode]);

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="text-[11px] uppercase tracking-wider text-muted-foreground">Histogram</span>
        <div className="flex rounded bg-muted p-0.5 gap-0.5">
          {(['luma', 'rgb', 'ev'] as const).map((m) => (
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
              {m === 'luma' ? 'L' : m === 'rgb' ? 'RGB' : 'EV'}
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
