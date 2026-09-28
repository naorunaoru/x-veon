import type { ExportHost } from '@/host';
import { encoderFor } from './encoders';
import { triggerDownload } from './download';
/** One worker owns one encode at a time; its event handlers cannot be overwritten by another job. */
let encoding: Promise<unknown> = Promise.resolve();
export function createExporter(deliver = triggerDownload): ExportHost {
  return {
    status: async () => ({ available: true }),
    chooseDestination: async (name) => ({ token: name }),
    encode: (job, destination) => {
      const operation = encoding
        .catch(() => {})
        .then(async () => {
          job.signal?.throwIfAborted();
          const blob = await encoderFor(job.format).encode(
            job.data,
            job.hdrData,
            job.width,
            job.height,
            job.orientation,
            job.quality,
            job.peakLuminance,
          );
          job.signal?.throwIfAborted();
          deliver(blob, destination.token);
          return { blob };
        });
      encoding = operation;
      return operation;
    },
  };
}
