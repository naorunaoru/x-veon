import type { ExportFormat } from '@/lib/types';
import { exportFormatInfo } from '@/lib/catalog';
import { encodeViaWorker } from './encoder';

export interface Encoder {
  format: ExportFormat;
  /**
   * `hdr` is the Rec.2020 HDR render for formats whose catalogue entry has `needsHdr`; null otherwise.
   * Both arrays are consumed: they are transferred to the encoder worker and detach.
   */
  encode(
    sdr: Float32Array,
    hdr: Float32Array | null,
    width: number,
    height: number,
    orientation: string,
    quality: number,
    peakLuminance: number,
  ): Promise<{ blob: Blob; encodeMs: number }>;
}

/** Every format is produced by the Rust encoder worker; the registry is still explicit per format. */
function workerEncoder(format: ExportFormat): Encoder {
  return {
    format,
    async encode(sdr, hdr, width, height, orientation, quality, peakLuminance) {
      const { bytes, encodeMs } = await encodeViaWorker(
        sdr,
        hdr ?? new Float32Array(0),
        width,
        height,
        orientation,
        format,
        quality,
        peakLuminance,
      );
      return { blob: new Blob([bytes.buffer as ArrayBuffer], { type: exportFormatInfo(format).mime }), encodeMs };
    },
  };
}

/** One encoder per catalogue format (lib/catalog.ts). index.test.ts holds the two lists equal. */
export const ENCODERS: readonly Encoder[] = [
  workerEncoder('jpeg-hdr'),
  workerEncoder('avif'),
  workerEncoder('tiff'),
];

export function encoderFor(format: ExportFormat): Encoder {
  const encoder = ENCODERS.find((candidate) => candidate.format === format);
  if (!encoder) throw new Error(`unknown export format: ${format}`);
  return encoder;
}
