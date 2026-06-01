import { useEffect, useRef, useState } from 'react';
import { useAppStore } from '@/store';
import type { HistogramChannel, HistogramMode } from '@/gl/renderer';
import './Scopes.css';
import './Panels.css';

type Source = 'scene' | 'display';

function toRendererMode(channel: HistogramChannel, source: Source): HistogramMode {
  const isLog = channel === 'ev';
  if (source === 'display') return isLog ? 'display-log' : 'display-linear';
  return isLog ? 'log' : 'linear';
}

const SOURCES: { id: Source; label: string }[] = [
  { id: 'display', label: 'Display' },
  { id: 'scene', label: 'Scene' },
];
const CHANNELS: { id: HistogramChannel; label: string }[] = [
  { id: 'luma', label: 'L' },
  { id: 'rgb', label: 'RGB' },
  { id: 'ev', label: 'EV' },
];

export function ScopeHistogram() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [channel, setChannel] = useState<HistogramChannel>('rgb');
  const [source, setSource] = useState<Source>('display');

  const renderer = useAppStore((s) => s.rendererRef);
  const selectedFile = useAppStore((s) => s.files.find((f) => f.id === s.selectedFileId));
  const lookPreset = selectedFile?.lookPreset ?? 'default';
  const overrides = selectedFile?.openDrtOverrides ?? {};
  const preProcess = selectedFile?.preProcessOverrides ?? {};

  // Attach/detach the histogram canvas to the GPU renderer.
  useEffect(() => {
    if (!renderer || !canvasRef.current) return;
    renderer.setHistogramCanvas(canvasRef.current);
    return () => { renderer.setHistogramCanvas(null); };
  }, [renderer]);

  // Update mode/channel and re-render when params or mode change.
  useEffect(() => {
    if (!renderer) return;
    renderer.histogramMode = toRendererMode(channel, source);
    renderer.histogramChannel = channel;
    renderer.render();
  }, [renderer, lookPreset, overrides, preProcess, channel, source]);

  return (
    <div className="xv-pgroup">
      <div className="xv-modes">
        <div className="xv-modes__grp">
          {SOURCES.map((s) => (
            <button
              key={s.id}
              className={`xv-modes__btn${source === s.id ? ' is-active' : ''}`}
              onClick={() => setSource(s.id)}
            >
              {s.label}
            </button>
          ))}
        </div>
        <div className="xv-modes__grp">
          {CHANNELS.map((c) => (
            <button
              key={c.id}
              className={`xv-modes__btn${channel === c.id ? ' is-active' : ''}`}
              onClick={() => setChannel(c.id)}
            >
              {c.label}
            </button>
          ))}
        </div>
      </div>
      <canvas ref={canvasRef} width={464} height={180} className="xv-scope-canvas" />
    </div>
  );
}
