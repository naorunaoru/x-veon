import { useEffect, useRef } from 'react';
import { useAppStore } from '@/app/store';
import { useHistogramCanvas } from '@/app/hooks/useHistogramCanvas';
import type { HistogramChannel, HistogramMode } from '@/renderer';
import './HistogramHud.css';
import { ScopeControls } from './ScopeControls';

type Source = 'scene' | 'display';

export function toRendererMode(channel: HistogramChannel, source: Source): HistogramMode {
  const isLog = channel === 'ev';
  if (source === 'display') return isLog ? 'display-log' : 'display-linear';
  return isLog ? 'log' : 'linear';
}

const CHANNEL_LABEL: Record<HistogramChannel, string> = { rgb: 'RGB', luma: 'LUMA', ev: 'EV' };

export function HistogramHud() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const renderer = useAppStore((s) => s.renderer);
  const source = useAppStore((s) => s.histogramSource);
  const channel = useAppStore((s) => s.histogramChannel);
  useHistogramCanvas(canvasRef);

  // Grade changes are already invalidated by OutputCanvas; only scope controls belong here.
  useEffect(() => {
    if (!renderer) return;
    renderer.histogram.setMode(toRendererMode(channel, source));
    renderer.histogram.setChannel(channel);
    renderer.requestRender();
  }, [renderer, channel, source]);

  return (
    <div className="xv-histhud xv-glass">
      <div className="xv-histhud__head">
        <span className="xv-histhud__label">{CHANNEL_LABEL[channel]}</span>
      </div>
      <canvas ref={canvasRef} width={464} height={84} className="xv-histhud__canvas" />
      <ScopeControls />
    </div>
  );
}
