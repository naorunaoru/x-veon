import { HdrRenderer } from './renderer';
import type { Renderer } from './renderer';
export type { Renderer, HistogramControls, DisplayGamut } from './renderer';
export type { HistogramMode, HistogramChannel } from './histogram';
export type { GradingConfig, TonescaleParams } from './grading/opendrt-params';

export function createRenderer(canvas: HTMLCanvasElement, opts?: { hdr?: boolean; headroom?: number }): Promise<Renderer> {
  return HdrRenderer.create(canvas, opts);
}
export function isWebGpuSupported(): boolean { return 'gpu' in navigator; }
