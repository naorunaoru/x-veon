import { useEffect, useRef, useState } from 'react';
import { cn } from '@/lib/utils';
import { useAppStore } from '@/store';
import type { HistogramChannel, HistogramMode } from '@/gl/renderer';

type Source = 'scene' | 'display';

function toRendererMode(channel: HistogramChannel, source: Source): HistogramMode {
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
  const [channel, setChannel] = useState<HistogramChannel>('rgb');
  const [source, setSource] = useState<Source>('display');

  const renderer = useAppStore((s) => s.rendererRef);
  const selectedFile = useAppStore((s) => s.files.find((f) => f.id === s.selectedFileId));
  const lookPreset = selectedFile?.lookPreset ?? 'default';
  const overrides = selectedFile?.openDrtOverrides ?? {};
  const preProcess = selectedFile?.preProcessOverrides ?? {};

  // Attach/detach the histogram canvas to the GPU renderer
  useEffect(() => {
    if (!renderer || !canvasRef.current) return;
    renderer.setHistogramCanvas(canvasRef.current);
    return () => { renderer.setHistogramCanvas(null); };
  }, [renderer]);

  // Update histogram mode/channel and re-render when params change
  useEffect(() => {
    if (!renderer) return;
    renderer.histogramMode = toRendererMode(channel, source);
    renderer.histogramChannel = channel;
    renderer.render();
  }, [renderer, lookPreset, overrides, preProcess, channel, source]);

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
