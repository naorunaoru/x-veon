import { useEffect, useRef, useState } from 'react';
import { cn } from '@/lib/utils';
import { useAppStore } from '@/store';
import type { HistogramData, HistogramMode } from '@/gl/renderer';

type Channel = 'luma' | 'rgb' | 'ev';
type Source = 'scene' | 'display';

const BINS = 256;
const EV_MIN = -8;
const EV_RANGE = 16;
const ZONE_COUNT = 16;
const ZONE_BAR_H = 16; // logical canvas pixels

// Linear histogram range: [0, LIN_MAX]. 1.0 = SDR clip point.
const LIN_MAX = 2.0;
const CLIP_BIN = Math.floor((1.0 / LIN_MAX) * (BINS - 1)); // bin index at 1.0

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

/** Tinted background + vertical clip line for the HDR region (>1.0) in linear modes. */
function drawHdrOverlay(ctx: CanvasRenderingContext2D, w: number, h: number): void {
  const clipX = (CLIP_BIN / (BINS - 1)) * w;

  // Tinted background for >1.0 region
  ctx.fillStyle = 'rgba(255, 140, 60, 0.08)';
  ctx.fillRect(clipX, 0, w - clipX, h);

  // Clip line at 1.0
  ctx.strokeStyle = 'rgba(255, 140, 60, 0.4)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(clipX, 0);
  ctx.lineTo(clipX, h);
  ctx.stroke();
}

function drawEvTicks(ctx: CanvasRenderingContext2D, w: number, h: number): void {
  const ticks = [-6, -3, 0, 3, 6];
  ctx.lineWidth = 1;
  ctx.font = '16px sans-serif';
  ctx.textAlign = 'center';

  for (const ev of ticks) {
    const x = ((ev - EV_MIN) / EV_RANGE) * w;
    ctx.strokeStyle = ev === 0 ? 'rgba(255, 255, 255, 0.35)' : 'rgba(255, 255, 255, 0.15)';
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
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
  channel: Channel,
): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const w = canvas.width;
  const fullH = canvas.height;
  const h = channel === 'ev' ? fullH - ZONE_BAR_H - 4 : fullH;
  ctx.clearRect(0, 0, w, fullH);

  // Find global max for consistent Y scaling
  let maxCount = 0;
  const channels = channel === 'rgb' ? [data.r, data.g, data.b] : [data.l];
  for (const ch of channels) {
    for (let i = 1; i < BINS - 1; i++) {
      if (ch[i] > maxCount) maxCount = ch[i];
    }
  }
  if (maxCount === 0) return;
  const maxLog = Math.log(1 + maxCount);

  if (channel === 'ev') {
    drawChannel(ctx, data.l, w, h, maxLog, 'rgba(120, 180, 255, 0.6)', 'rgba(160, 210, 255, 0.5)');
    drawEvTicks(ctx, w, h);
    drawZoneBar(ctx, data.l, w, h + 4);
  } else if (channel === 'luma') {
    drawHdrOverlay(ctx, w, h);
    drawChannel(ctx, data.l, w, h, maxLog, 'rgba(200, 200, 200, 0.7)', 'rgba(255, 255, 255, 0.4)');
  } else {
    drawHdrOverlay(ctx, w, h);
    ctx.globalCompositeOperation = 'screen';
    drawChannel(ctx, data.r, w, h, maxLog, 'rgba(220, 50, 50, 0.6)', 'rgba(255, 80, 80, 0.5)');
    drawChannel(ctx, data.g, w, h, maxLog, 'rgba(50, 200, 50, 0.6)', 'rgba(80, 255, 80, 0.5)');
    drawChannel(ctx, data.b, w, h, maxLog, 'rgba(50, 80, 220, 0.6)', 'rgba(80, 120, 255, 0.5)');
    ctx.globalCompositeOperation = 'source-over';
  }
}

function toRendererMode(channel: Channel, source: Source): HistogramMode {
  const isLog = channel === 'ev';
  if (source === 'display') return isLog ? 'display-log' : 'display-linear';
  return isLog ? 'log' : 'linear';
}

const toggleBtnClass = (active: boolean) => cn(
  'px-1.5 py-0 text-[10px] font-medium rounded-sm transition-colors',
  active
    ? 'bg-background text-foreground shadow-sm'
    : 'text-muted-foreground hover:text-foreground',
);

export function Histogram() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [channel, setChannel] = useState<Channel>('luma');
  const [source, setSource] = useState<Source>('scene');

  const renderer = useAppStore((s) => s.rendererRef);
  const selectedFile = useAppStore((s) => s.files.find((f) => f.id === s.selectedFileId));
  const lookPreset = selectedFile?.lookPreset ?? 'default';
  const overrides = selectedFile?.openDrtOverrides ?? {};

  useEffect(() => {
    if (!renderer || !canvasRef.current) return;

    renderer.histogramMode = toRendererMode(channel, source);

    let cancelled = false;

    renderer.render();
    renderer.getHistogramData().then((data) => {
      if (!cancelled && data && canvasRef.current) {
        drawHistogram(canvasRef.current, data, channel);
      }
    });

    return () => { cancelled = true; };
  }, [renderer, lookPreset, overrides, channel, source]);

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <div className="flex rounded bg-muted p-0.5 gap-0.5">
          {(['scene', 'display'] as const).map((s) => (
            <button
              key={s}
              className={toggleBtnClass(source === s)}
              onClick={() => setSource(s)}
            >
              {s === 'scene' ? 'Scene' : 'Display'}
            </button>
          ))}
        </div>
        <div className="flex rounded bg-muted p-0.5 gap-0.5">
          {(['luma', 'rgb', 'ev'] as const).map((m) => (
            <button
              key={m}
              className={toggleBtnClass(channel === m)}
              onClick={() => setChannel(m)}
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
