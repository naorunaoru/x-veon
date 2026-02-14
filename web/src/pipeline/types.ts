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

export interface RotatedImage {
  data: Float32Array;
  width: number;
  height: number;
}

export type DemosaicMethod = 'neural-net' | 'markesteijn1' | 'bilinear' | 'dht';

export type ExportFormat = 'avif' | 'jpeg' | 'tiff';

export interface ExportData {
  hwc: Float32Array;
  width: number;
  height: number;
  xyzToCam: Float32Array | null;
  orientation: string;
}

export interface ProcessingResult {
  imageData: ImageData;
  exportData: ExportData;
  isHdr: boolean;
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
