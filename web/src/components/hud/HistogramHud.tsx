import { useEffect, useRef } from 'react';
import { useAppStore } from '@/store';
import { useHistogramCanvas } from '@/hooks/useHistogramCanvas';
import type { HistogramChannel, HistogramMode } from '@/gl/renderer';
import type { OpenDrtConfig, PreProcessConfig } from '@/gl/opendrt-params';
import './HistogramHud.css';

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
  const renderer = useAppStore((s) => s.rendererRef);
  const source = useAppStore((s) => s.histogramSource);
  const channel = useAppStore((s) => s.histogramChannel);
  const selectedFile = useAppStore((s) => s.files.find((f) => f.id === s.selectedFileId));
  const overrides = selectedFile?.openDrtOverrides ?? EMPTY_OVERRIDES;
  const preProcess = selectedFile?.preProcessOverrides ?? EMPTY_PREPROCESS;
  const lookPreset = selectedFile?.lookPreset ?? 'default';

  useHistogramCanvas(canvasRef);

  // The always-mounted widget owns mode/channel sync + re-render on grade change;
  // render() fans out to every registered canvas (this widget + the Scopes panel).
  // ScopesPanel relies on this: it lives beside this widget under HudRoot's hasFiles
  // branch, so this effect drives the panel's live updates too (the panel only does
  // its own initial paint, via useHistogramCanvas).
  useEffect(() => {
    if (!renderer) return;
    renderer.histogramMode = toRendererMode(channel, source);
    renderer.histogramChannel = channel;
    renderer.render();
  }, [renderer, lookPreset, overrides, preProcess, channel, source]);

  return (
    <div className="xv-histhud xv-glass">
      <div className="xv-histhud__head">
        <span className="xv-histhud__label">{CHANNEL_LABEL[channel]}</span>
      </div>
      <canvas ref={canvasRef} width={464} height={84} className="xv-histhud__canvas" />
    </div>
  );
}
