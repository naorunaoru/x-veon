import { useEffect, useRef } from 'react';
import { useAppStore } from '@/app/store';
import { useHistogramCanvas } from '@/app/hooks/useHistogramCanvas';
import type { HistogramChannel, HistogramMode } from '@/renderer';
import type { OpenDrtConfig, PreProcessConfig } from '@/renderer/grading/opendrt-params';
import './HistogramHud.css';
import { ScopeControls } from './ScopeControls';

type Source = 'scene' | 'display';

// Stable fallbacks so an ungraded file keeps the same override refs across renders —
// otherwise `?? {}` makes fresh objects that needlessly re-fire the render effect
// (and now each render() fans out to every histogram canvas). Mirrors OutputCanvas.
const EMPTY_OVERRIDES: Partial<OpenDrtConfig> = {};
const EMPTY_PREPROCESS: Partial<PreProcessConfig> = {};

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
  const selectedFile = useAppStore((s) => s.files.find((f) => f.id === s.selectedFileId));
  const overrides = selectedFile?.openDrtOverrides ?? EMPTY_OVERRIDES;
  const preProcess = selectedFile?.preProcessOverrides ?? EMPTY_PREPROCESS;
  const lookPreset = selectedFile?.lookPreset ?? 'default';

  useHistogramCanvas(canvasRef);

  // The widget owns histogram controls and redraws as the selected grade changes.
  useEffect(() => {
    if (!renderer) return;
    renderer.histogram.setMode(toRendererMode(channel, source));
    renderer.histogram.setChannel(channel);
    renderer.render();
  }, [renderer, lookPreset, overrides, preProcess, channel, source]);

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
