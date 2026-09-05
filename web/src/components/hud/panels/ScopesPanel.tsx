import { useRef } from 'react';
import { FloatingPanel } from '../FloatingPanel';
import { useAppStore } from '@/app/store';
import { useHistogramCanvas } from '@/app/hooks/useHistogramCanvas';
import type { HistogramChannel } from '@/renderer';
import './Scopes.css';
import './Panels.css';

type Source = 'scene' | 'display';
const SOURCES: { id: Source; label: string }[] = [
  { id: 'display', label: 'Display' },
  { id: 'scene', label: 'Scene' },
];
const CHANNELS: { id: HistogramChannel; label: string }[] = [
  { id: 'luma', label: 'L' },
  { id: 'rgb', label: 'RGB' },
  { id: 'ev', label: 'EV' },
];

export function ScopesPanel() {
  const setOpenPanel = useAppStore((s) => s.setOpenPanel);
  const source = useAppStore((s) => s.histogramSource);
  const channel = useAppStore((s) => s.histogramChannel);
  const setSource = useAppStore((s) => s.setHistogramSource);
  const setChannel = useAppStore((s) => s.setHistogramChannel);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useHistogramCanvas(canvasRef);

  return (
    <FloatingPanel title="Scopes" onClose={() => setOpenPanel(null)}>
      <div className="xv-pgroup">
        <span className="xv-pgroup__title">Histogram</span>
        <canvas ref={canvasRef} width={464} height={180} className="xv-scope-canvas" />
        <div className="xv-modes">
          <div className="xv-modes__grp">
            {SOURCES.map((s) => (
              <button key={s.id} className={`xv-modes__btn${source === s.id ? ' is-active' : ''}`}
                onClick={() => setSource(s.id)}>{s.label}</button>
            ))}
          </div>
          <div className="xv-modes__grp">
            {CHANNELS.map((c) => (
              <button key={c.id} className={`xv-modes__btn${channel === c.id ? ' is-active' : ''}`}
                onClick={() => setChannel(c.id)}>{c.label}</button>
            ))}
          </div>
        </div>
      </div>
    </FloatingPanel>
  );
}
