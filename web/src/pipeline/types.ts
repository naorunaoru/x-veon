import type { CfaType } from '@/lib/types';

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

/** A decoded readout without its photosite data (what the decode worker sends back). */
export type RawMeta = Omit<RawImage, 'data'>;

/** A rectangle within an image. */
export interface Rect {
  left: number;
  top: number;
  width: number;
  height: number;
}

/** The visible CFA, padded so its phase matches the reference pattern. Raw u16 values. */
export interface PaddedCfa {
  data: Uint16Array;
  width: number;
  height: number;
  padTop: number;
  padLeft: number;
}

/** A padded CFA plus its layout, calibrated white levels and normalisation table. */
export interface PreparedCfa extends PaddedCfa {
  visibleWidth: number;
  visibleHeight: number;
  cfa: CfaInfo;
  /** Calibrated white levels, RGBE order. */
  whiteLevels: Uint16Array;
  /** `lut[colour * NORM_LUT_SIZE + value]`: the normalised float32 value (see normalizationLut). */
  lut: Float32Array;
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

