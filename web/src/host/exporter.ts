import type { ExportHost } from '@/host';
import { encoderFor } from './encoders';
import { triggerDownload } from './download';
/** One worker owns one encode at a time; its event handlers cannot be overwritten by another job. */
let encoding: Promise<unknown> = Promise.resolve();
export function createExporter(deliver = triggerDownload): ExportHost {
  return {
    status: async () => ({ available: true }),
    resolveTarget: async () => null,
    encode: (job, out) => {
      const operation = encoding
        .catch(() => {})
        .then(async () => {
          job.signal?.throwIfAborted();
          const { blob, encodeMs } = await encoderFor(job.format).encode(
            job.data,
            job.hdrData,
            job.width,
            job.height,
            job.orientation,
            job.quality,
            job.peakLuminance,
          );
          job.signal?.throwIfAborted();
          deliver(blob, out.name);
          return { outcome: 'written' as const, blob, bytes: blob.size, name: out.name, encodeMs };
        });
      encoding = operation;
      return operation;
    },
  };
}
