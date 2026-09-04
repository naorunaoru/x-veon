import type { CfaType, DemosaicMethod } from '@/lib/types';
import type { ChannelMasks } from '../types';
import type { PipelineContext, ProcessOptions } from '../context';
import { cropToHWC } from '../postprocess/postprocessor';

export type TraditionalMethod = Exclude<DemosaicMethod, 'neural-net'>;

/** The padded, canonically aligned CFA plus what a strategy needs to crop its output. */
export interface DemosaicInput {
  cfa: Float32Array;
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

/** Cropped HWC RGB at the visible size: on the GPU for the neural net, on the CPU otherwise. */
export interface DemosaicOutput {
  hwc: Float32Array | GPUBuffer;
  tileCount: number;
}

export interface DemosaicStrategy {
  id: DemosaicMethod;
  run(input: DemosaicInput, ctx: PipelineContext, opts: ProcessOptions): Promise<DemosaicOutput>;
}

/** Crop a planar (CHW, padded) demosaic result to the visible HWC image. */
export function cropPlanar(planar: Float32Array, input: DemosaicInput): DemosaicOutput {
  return {
    hwc: cropToHWC(
      planar, input.height, input.width,
      input.padTop, input.padLeft, input.visibleHeight, input.visibleWidth,
    ),
    tileCount: 1,
  };
}
