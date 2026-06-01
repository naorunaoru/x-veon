import { useEffect, useRef } from 'react';
import { useAppStore } from '@/store';
import type { HistogramChannel, HistogramMode } from '@/gl/renderer';
import './HistogramHud.css';

type Source = 'scene' | 'display';

export function toRendererMode(channel: HistogramChannel, source: Source): HistogramMode {
  const isLog = channel === 'ev';
  if (source === 'display') return isLog ? 'display-log' : 'display-linear';
  return isLog ? 'log' : 'linear';
}

const CHANNEL_LABEL: Record<HistogramChannel, string> = { rgb: 'RGB', luma: 'LUMA', ev: 'EV' };

export function HistogramHud() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const renderer = useAppStore((s) => s.rendererRef);
  const source = useAppStore((s) => s.histogramSource);
  const channel = useAppStore((s) => s.histogramChannel);
  const selectedFile = useAppStore((s) => s.files.find((f) => f.id === s.selectedFileId));
  const overrides = selectedFile?.openDrtOverrides ?? {};
  const preProcess = selectedFile?.preProcessOverrides ?? {};
  const lookPreset = selectedFile?.lookPreset ?? 'default';

  useEffect(() => {
    if (!renderer || !canvasRef.current) return;
    renderer.setHistogramCanvas(canvasRef.current);
    return () => { renderer.setHistogramCanvas(null); };
  }, [renderer]);

  useEffect(() => {
    if (!renderer) return;
    renderer.histogramMode = toRendererMode(channel, source);
    renderer.histogramChannel = channel;
    renderer.render();
  }, [renderer, lookPreset, overrides, preProcess, channel, source]);

  const range = channel === 'ev' ? '-8 — +8' : '0 — 1.0';

  return (
    <div className="xv-histhud xv-glass">
      <div className="xv-histhud__head">
        <span className="xv-histhud__label">{CHANNEL_LABEL[channel]}</span>
        <span>{range}</span>
      </div>
      <canvas ref={canvasRef} width={464} height={84} className="xv-histhud__canvas" />
    </div>
  );
}
