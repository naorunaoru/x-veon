export interface RawImage {
  data: Uint16Array;
  width: number;
  height: number;
  wbCoeffs: Float32Array;
  blackLevels: Uint16Array;
  whiteLevels: Uint16Array;
  xyzToCam: Float32Array;
  camToXyz: Float32Array;
  orientation: string;
  make: string;
  model: string;
  cfaStr: string;
  cfaWidth: number;
  crops: Uint16Array;
  drGain: number;
  exposureBias: number;
  lensModel: string;
  focalLength: number;
  fNumber: number;
}

export interface CroppedImage {
  data: Uint16Array;
  width: number;
  height: number;
}

export interface PaddedImage {
  data: Float32Array;
  width: number;
  height: number;
  padTop: number;
  padLeft: number;
}

export interface CfaInfo {
  cfaType: CfaType;
  pattern: readonly (readonly number[])[];
  period: number;
  dy: number;
  dx: number;
}

export interface PatternShift {
  dy: number;
  dx: number;
}

export interface TileGrid {
  tiles: Array<{ x: number; y: number }>;
  hPad: number;
  wPad: number;
}

export interface ChannelMasks {
  r: Float32Array;
  g: Float32Array;
  b: Float32Array;
}

import type { CfaType } from '@/lib/types';
