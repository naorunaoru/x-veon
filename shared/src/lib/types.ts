export type CfaType = 'xtrans' | 'bayer';

export type DemosaicMethod =
  | 'neural-net'
  | 'bilinear'
  | 'markesteijn3'
  | 'markesteijn1'
  | 'dht'
  | 'ahd'
  | 'ppg'
  | 'mhc';

export type ModelSize = 'S' | 'M' | 'L';
export interface ModelIdentity { size: ModelSize; sha256: string }
export type ExportFormat = 'jpeg-hdr' | 'avif' | 'tiff';
export type LookPreset = 'opendrt-v1-default' | 'opendrt-v1-colorful' | 'opendrt-v1-umbra' | 'opendrt-v1-base'
  | 'default' | 'colorful' | 'umbra' | 'base' | 'flat'
  | 'low-contrast' | 'medium-contrast' | 'aces-2' | 'marvelous';

/** What the app stores and shows about a processed file (produced by the pipeline). */
export interface ProcessingMetadata {
  make: string;
  model: string;
  width: number;
  height: number;
  tileCount: number;
  inferenceTime: number;
  backend: string;
  exposureBias: number;
  lensModel: string;
  focalLength: number;
  fNumber: number;
  colorTemp: number;
  tint: number;
  modelSize?: ModelSize;
  modelIdentity?: ModelIdentity;
  cfaType?: CfaType;
  modelNote?: string | null;
}

/** Lightweight export data stored in Zustand; pixel data lives in OPFS. */
export interface ExportDataMeta {
  width: number;
  height: number;
  xyzToCam: Float32Array | null;
  wbCoeffs: Float32Array;
  camToXyz: Float32Array;
  orientation: string;
}

export interface ProcessingResultMeta {
  exportData: ExportDataMeta;
  metadata: ProcessingMetadata;
}

/** IDB-safe version of ProcessingResultMeta (Float32Array → number[]). */
export interface SerializableResultMeta {
  exportData: {
    width: number;
    height: number;
    xyzToCam: number[] | null;
    wbCoeffs: number[];
    camToXyz?: number[];
    orientation: string;
  };
  metadata: ProcessingMetadata;
}

export function serializeResultMeta(meta: ProcessingResultMeta): SerializableResultMeta {
  return {
    exportData: {
      width: meta.exportData.width,
      height: meta.exportData.height,
      xyzToCam: meta.exportData.xyzToCam ? Array.from(meta.exportData.xyzToCam) : null,
      wbCoeffs: Array.from(meta.exportData.wbCoeffs),
      camToXyz: Array.from(meta.exportData.camToXyz),
      orientation: meta.exportData.orientation,
    },
    metadata: meta.metadata,
  };
}

export function deserializeResultMeta(meta: SerializableResultMeta): ProcessingResultMeta {
  // Default cam_to_xyz: 3×4 identity fallback for data persisted before this field existed.
  // Layout is row-major with stride 4 (4th column = unused camera channel).
  const camToXyz = meta.exportData.camToXyz
    ? new Float32Array(meta.exportData.camToXyz)
    : new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]);
  return {
    exportData: {
      width: meta.exportData.width,
      height: meta.exportData.height,
      xyzToCam: meta.exportData.xyzToCam ? new Float32Array(meta.exportData.xyzToCam) : null,
      wbCoeffs: new Float32Array(meta.exportData.wbCoeffs),
      camToXyz,
      orientation: meta.exportData.orientation,
    },
    metadata: meta.metadata,
  };
}

/** RGBA32F image resident on the GPU; produced by the pipeline, consumed by the renderer.
 * Alpha carries the clamped clip ratio (max channel / clip level) from GPU highlight recovery. */
export interface GpuImage {
  /** rgba32float, TEXTURE_BINDING | COPY_SRC; the renderer samples it without copying. */
  texture: GPUTexture;
  width: number;
  height: number;
}
