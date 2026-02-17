export interface RawImage {
  data: Uint16Array;
  width: number;
  height: number;
  wbCoeffs: Float32Array;
  blackLevels: Uint16Array;
  whiteLevels: Uint16Array;
  xyzToCam: Float32Array;
  orientation: string;
  make: string;
  model: string;
  cfaStr: string;
  cfaWidth: number;
  crops: Uint16Array;
  drGain: number;
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

export type CfaType = 'xtrans' | 'bayer';

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
  paddedCfa: Float32Array;
  hPad: number;
  wPad: number;
}

export interface ChannelMasks {
  r: Float32Array;
  g: Float32Array;
  b: Float32Array;
}

export type DemosaicMethod =
  | 'neural-net'
  | 'bilinear'
  // X-Trans
  | 'markesteijn3'
  | 'markesteijn1'
  | 'dht'
  // Bayer
  | 'ahd'
  | 'ppg'
  | 'mhc';

export type ExportFormat = 'jpeg-hdr' | 'avif' | 'tiff';

export type LookPreset = 'default' | 'base' | 'flat';

export interface ExportData {
  hwc: Float32Array;
  width: number;
  height: number;
  xyzToCam: Float32Array | null;
  wbCoeffs: Float32Array;
  orientation: string;
}

export interface ProcessingResult {
  exportData: ExportData;
  metadata: {
    make: string;
    model: string;
    width: number;
    height: number;
    tileCount: number;
    inferenceTime: number;
    backend: string;
  };
}

/** Lightweight result stored in Zustand — hwc pixel data lives in OPFS. */
export interface ExportDataMeta {
  width: number;
  height: number;
  xyzToCam: Float32Array | null;
  wbCoeffs: Float32Array;
  orientation: string;
}

export interface ProcessingResultMeta {
  exportData: ExportDataMeta;
  metadata: ProcessingResult['metadata'];
}

/** IDB-safe version of ProcessingResultMeta (Float32Array → number[]). */
export interface SerializableResultMeta {
  exportData: {
    width: number;
    height: number;
    xyzToCam: number[] | null;
    wbCoeffs: number[];
    orientation: string;
  };
  metadata: ProcessingResult['metadata'];
}

export function serializeResultMeta(meta: ProcessingResultMeta): SerializableResultMeta {
  return {
    exportData: {
      width: meta.exportData.width,
      height: meta.exportData.height,
      xyzToCam: meta.exportData.xyzToCam ? Array.from(meta.exportData.xyzToCam) : null,
      wbCoeffs: Array.from(meta.exportData.wbCoeffs),
      orientation: meta.exportData.orientation,
    },
    metadata: meta.metadata,
  };
}

export function deserializeResultMeta(meta: SerializableResultMeta): ProcessingResultMeta {
  return {
    exportData: {
      width: meta.exportData.width,
      height: meta.exportData.height,
      xyzToCam: meta.exportData.xyzToCam ? new Float32Array(meta.exportData.xyzToCam) : null,
      wbCoeffs: new Float32Array(meta.exportData.wbCoeffs),
      orientation: meta.exportData.orientation,
    },
    metadata: meta.metadata,
  };
}
