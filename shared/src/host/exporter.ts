import type { ExportFormat } from '@/lib/types';
/** Token issued and interpreted only by the host. */
export interface ExportDestination {
  readonly token: string;
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
  blob?: Blob;
}
export interface ExportHost {
  status(): Promise<{ available: true } | { available: false; reason: string }>;
  chooseDestination(suggestedName: string, format: ExportFormat): Promise<ExportDestination | null>;
  encode(job: EncodeJob, destination: ExportDestination): Promise<ExportResult>;
}
