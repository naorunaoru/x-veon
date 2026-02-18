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
const LIN_MAX = 1.2;
const CLIP_BIN = Math.floor((1.0 / LIN_MAX) * (BINS - 1)); // bin index at 1.0

interface BinRange {
  lo: number;
  hi: number;
}

/** Scan bins to find the actual occupied range. */
function findBinRange(channels: Uint32Array[], forceZeroLo: boolean): BinRange {
  let lo = BINS;
  let hi = -1;
  for (const bins of channels) {
    for (let i = 0; i < BINS; i++) {
      if (bins[i] > 0) {
        if (i < lo) lo = i;
        if (i > hi) hi = i;
      }
    }
  }
  if (hi < lo) return { lo: 0, hi: BINS - 1 }; // empty → full range
  const span = Math.max(hi - lo, 1);
  const pad = Math.max(2, Math.round(span * 0.05));
  lo = Math.max(0, lo - pad);
  hi = Math.min(BINS - 1, hi + pad);
  if (forceZeroLo) lo = 0;
  return { lo, hi };
}

/** Convert a bin index to an x coordinate within the canvas. */
function binToX(bin: number, range: BinRange, w: number): number {
  return ((bin - range.lo) / (range.hi - range.lo)) * w;
}

function drawChannel(
  ctx: CanvasRenderingContext2D,
  bins: Uint32Array,
  w: number,
  h: number,
  maxLog: number,
  fillColor: string,
  strokeColor: string,
  range: BinRange,
) {
  const { lo, hi } = range;
  const span = hi - lo;
  ctx.beginPath();
  ctx.moveTo(0, h);
  for (let i = lo; i <= hi; i++) {
    const x = ((i - lo) / span) * w;
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
function drawHdrOverlay(ctx: CanvasRenderingContext2D, w: number, h: number, range: BinRange): void {
  if (CLIP_BIN < range.lo || CLIP_BIN > range.hi) return;
  const clipX = binToX(CLIP_BIN, range, w);

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

/** Choose nice EV tick spacing based on the visible range. */
function niceEvTicks(loEv: number, hiEv: number): number[] {
  const span = hiEv - loEv;
  let step: number;
  if (span <= 4) step = 0.5;
  else if (span <= 8) step = 1;
  else if (span <= 16) step = 2;
  else step = 3;

  const ticks: number[] = [];
  const start = Math.ceil(loEv / step) * step;
  for (let ev = start; ev <= hiEv + 0.001; ev += step) {
    ticks.push(Math.round(ev / step) * step);
  }
  return ticks;
}

function drawEvTicks(ctx: CanvasRenderingContext2D, w: number, h: number, range: BinRange): void {
  const loEv = EV_MIN + (range.lo / (BINS - 1)) * EV_RANGE;
  const hiEv = EV_MIN + (range.hi / (BINS - 1)) * EV_RANGE;
  const ticks = niceEvTicks(loEv, hiEv);

  ctx.lineWidth = 1;
  ctx.font = '16px sans-serif';
  ctx.textAlign = 'center';

  for (const ev of ticks) {
    const bin = ((ev - EV_MIN) / EV_RANGE) * (BINS - 1);
    const x = binToX(bin, range, w);
    ctx.strokeStyle = Math.abs(ev) < 0.01 ? 'rgba(255, 255, 255, 0.35)' : 'rgba(255, 255, 255, 0.15)';
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
    ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
    const label = Math.abs(ev) < 0.01 ? '0' : (ev > 0 ? `+${ev}` : `${ev}`);
    ctx.fillText(label, x, h - 4);
  }
}

function drawZoneBar(
  ctx: CanvasRenderingContext2D,
  bins: Uint32Array,
  w: number,
  y: number,
  range: BinRange,
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

  for (let z = 0; z < ZONE_COUNT; z++) {
    const zLo = z * 16;
    const zHi = (z + 1) * 16 - 1;
    if (zHi < range.lo || zLo > range.hi) continue;

    const xLo = Math.max(0, binToX(zLo, range, w));
    const xHi = Math.min(w, binToX(zHi + 1, range, w));
    const zW = xHi - xLo;
    if (zW < 1) continue;

    const t = z / (ZONE_COUNT - 1);
    const brightness = Math.round(30 + t * 200);
    const alpha = 0.15 + 0.85 * (zoneCounts[z] / maxZone);
    ctx.fillStyle = `rgba(${brightness}, ${brightness}, ${brightness}, ${alpha})`;
    ctx.fillRect(xLo, y, zW - 1, ZONE_BAR_H);
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

  // Determine visible bin range from actual data
  const rangeChannels = channel === 'rgb' ? [data.r, data.g, data.b] : [data.l];
  const range = findBinRange(rangeChannels, channel !== 'ev');

  // Find max count within visible range for Y scaling (skip extreme bins 0/255)
  let maxCount = 0;
  const channels = channel === 'rgb' ? [data.r, data.g, data.b] : [data.l];
  for (const ch of channels) {
    for (let i = range.lo; i <= range.hi; i++) {
      if (i === 0 || i === BINS - 1) continue;
      if (ch[i] > maxCount) maxCount = ch[i];
    }
  }
  if (maxCount === 0) return;
  const maxLog = Math.log(1 + maxCount);

  if (channel === 'ev') {
    drawChannel(ctx, data.l, w, h, maxLog, 'rgba(120, 180, 255, 0.6)', 'rgba(160, 210, 255, 0.5)', range);
    drawEvTicks(ctx, w, h, range);
    drawZoneBar(ctx, data.l, w, h + 4, range);
  } else if (channel === 'luma') {
    drawHdrOverlay(ctx, w, h, range);
    drawChannel(ctx, data.l, w, h, maxLog, 'rgba(200, 200, 200, 0.7)', 'rgba(255, 255, 255, 0.4)', range);
  } else {
    drawHdrOverlay(ctx, w, h, range);
    ctx.globalCompositeOperation = 'screen';
    drawChannel(ctx, data.r, w, h, maxLog, 'rgba(220, 50, 50, 0.6)', 'rgba(255, 80, 80, 0.5)', range);
    drawChannel(ctx, data.g, w, h, maxLog, 'rgba(50, 200, 50, 0.6)', 'rgba(80, 255, 80, 0.5)', range);
    drawChannel(ctx, data.b, w, h, maxLog, 'rgba(50, 80, 220, 0.6)', 'rgba(80, 120, 255, 0.5)', range);
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
  const [channel, setChannel] = useState<Channel>('rgb');
  const [source, setSource] = useState<Source>('display');

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
          {(['display', 'scene'] as const).map((s) => (
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
        height={180}
        className="w-full rounded-sm border border-white/15"
        style={{ background: 'rgba(0,0,0,0.3)', height: 90 }}
      />
    </div>
  );
}
