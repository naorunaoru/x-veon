import type { CfaType, DemosaicMethod, ExportFormat, ModelSize } from './types';

export interface DemosaicMethodInfo {
  id: DemosaicMethod;
  label: string;
  cfa: 'xtrans' | 'bayer' | 'any';
}

/** Every demosaic method the app offers, in Settings order. */
export const DEMOSAIC_METHODS: readonly DemosaicMethodInfo[] = [
  { id: 'neural-net', label: 'X-veon', cfa: 'any' },
  { id: 'markesteijn3', label: 'Markesteijn (3-pass)', cfa: 'xtrans' },
  { id: 'markesteijn1', label: 'Markesteijn (1-pass)', cfa: 'xtrans' },
  { id: 'dht', label: 'DHT', cfa: 'xtrans' },
  { id: 'ahd', label: 'AHD', cfa: 'bayer' },
  { id: 'ppg', label: 'PPG', cfa: 'bayer' },
  { id: 'mhc', label: 'MHC', cfa: 'bayer' },
  { id: 'bilinear', label: 'Bilinear', cfa: 'any' },
];

export function demosaicMethodsFor(cfa: CfaType | null): DemosaicMethodInfo[] {
  return DEMOSAIC_METHODS.filter((method) => method.cfa === 'any' || cfa === null || method.cfa === cfa);
}

export function isMethodValidForCfa(method: DemosaicMethod, cfa: CfaType): boolean {
  const info = DEMOSAIC_METHODS.find((candidate) => candidate.id === method);
  return !!info && (info.cfa === 'any' || info.cfa === cfa);
}

export interface ExportFormatInfo {
  id: ExportFormat;
  label: string;
  ext: string;
  mime: string;
  needsHdr: boolean;
}

/** Export formats in dialog order. `needsHdr`: the encoder also wants an HDR render. */
export const EXPORT_FORMATS: readonly ExportFormatInfo[] = [
  { id: 'jpeg-hdr', label: 'Ultra HDR JPEG', ext: 'jpg', mime: 'image/jpeg', needsHdr: true },
  { id: 'avif', label: 'AVIF (BT.2020 / HLG)', ext: 'avif', mime: 'image/avif', needsHdr: true },
  { id: 'tiff', label: 'TIFF (Linear sRGB)', ext: 'tif', mime: 'image/tiff', needsHdr: false },
];

export function exportFormatInfo(id: ExportFormat): ExportFormatInfo {
  const info = EXPORT_FORMATS.find((candidate) => candidate.id === id);
  if (!info) throw new Error(`unknown export format: ${id}`);
  return info;
}

export const MODEL_SIZES: readonly ModelSize[] = ['S', 'M', 'L'];

/** RAW inputs accepted by file pickers and the library service. */
export const RAW_EXTENSIONS = [
  '.raf', '.cr2', '.cr3', '.nef', '.nrw', '.arw', '.dng',
  '.rw2', '.orf', '.pef', '.srw', '.erf', '.kdc', '.dcr', '.mef',
];
export const RAW_ACCEPT = RAW_EXTENSIONS.join(',');
