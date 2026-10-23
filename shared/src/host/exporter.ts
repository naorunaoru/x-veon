import type { PhotoId } from './library';
import type { ExportFormat } from '@/lib/types';
/** Token issued and interpreted only by the host. */
export interface ExportTarget {
  readonly token: string;
}
export type SaveTo = { mode: 'beside' } | { mode: 'folder'; path: string };
export type OnExists = 'skip' | 'replace';
export interface ExportOutput {
  target: ExportTarget | null;
  name: string;
  onExists: OnExists;
}
export interface EncodeJob {
  format: ExportFormat;
  data: Float32Array;
  hdrData: Float32Array | null;
  width: number;
  height: number;
  orientation: string;
  quality: number;
  peakLuminance: number;
  signal?: AbortSignal;
}
export interface ExportResult {
  outcome: 'written' | 'skipped';
  blob?: Blob;
  bytes?: number;
  sha256?: string;
  encodeMs?: number;
  name?: string;
}
export interface ExportReveal {
  label: string;
  open(target: ExportTarget, name?: string): Promise<void>;
}
export interface ExportHost {
  status(): Promise<{ available: true } | { available: false; reason: string }>;
  /** One target per batch; web returns null and downloads each output by name. */
  resolveTarget(saveTo: SaveTo, photoIds: PhotoId[]): Promise<ExportTarget | null>;
  encode(job: EncodeJob, out: ExportOutput): Promise<ExportResult>;
  reveal?: ExportReveal;
  chooseFolder?(): Promise<string | null>;
}
