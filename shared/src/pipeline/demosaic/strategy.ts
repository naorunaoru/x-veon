import type { CfaType, DemosaicMethod } from '@/lib/types';
import type { ChannelMasks } from '../types';
import type { PipelineContext, ProcessOptions } from '../context';

export type TraditionalMethod = Exclude<DemosaicMethod, 'neural-net'>;

/** The padded, canonically aligned CFA plus what a strategy needs to crop its output. */
export interface DemosaicInput {
  /** Raw u16 photosite values, padded and phase-aligned; normalised through `lut` when consumed. */
  cfa: Uint16Array;
  /** Per-colour normalisation table (see normalizationLut). */
  lut: Float32Array;
  width: number;            // padded
  height: number;           // padded
  padTop: number;
  padLeft: number;
  visibleWidth: number;
  visibleHeight: number;
  pattern: Uint32Array;     // period × period, row-major, R=0 G=1 B=2
  period: number;
  cfaType: CfaType;
  masks: ChannelMasks;      // per-channel patch masks for the neural net
  clipNorm: [number, number, number];
}

/**
 * Demosaiced RGB on the GPU: HWC float32, `stride` pixels per row. The visible image starts at
 * (offsetX, offsetY), so a padded demosaic buffer is consumed in place without a crop copy.
 */
export interface GpuRgb {
  buffer: GPUBuffer;
  stride: number;
  offsetX: number;
  offsetY: number;
}

/** Visible-size RGB: on the GPU for the GPU methods, cropped HWC on the CPU for the WASM pool. */
export interface DemosaicOutput {
  rgb: Float32Array | GpuRgb;
  tileCount: number;
}

export interface DemosaicStrategy {
  id: DemosaicMethod;
  run(input: DemosaicInput, ctx: PipelineContext, opts: ProcessOptions): Promise<DemosaicOutput>;
}
