import { useAppStore } from '@/app/store';
import type { QueuedFile } from '@/app/store';
import { exportFormatInfo } from '@/lib/catalog';
import {
  configFromPreset,
  configWithOverrides,
  deriveHdrConfig,
  computeTonescaleParams,
} from '@/renderer/grading/opendrt-params';
import { createRenderer, type Renderer } from '@/renderer';
import type { ExportFormat } from '@/lib/types';
import type { EncodeJob, ExportResult, ExportDestination } from '@/host';
import { acquireResult } from './processing';
import { processingKey } from '@/app/store/photo';
import { getHost } from './host';
const HDR_PEAK_LUMINANCE = 1000;
export type ExportState = 'queued' | 'rendering' | 'encoding' | 'done' | 'failed' | 'cancelled';
export interface ExportObserver {
  state?(state: ExportState): void;
  destination?(destination: ExportDestination): void;
}
export interface ExportJobHandle {
  readonly state: ExportState;
  promise: Promise<ExportResult | null>;
  cancel(): void;
}
// A slot owns one export's rendered planes until the host finishes encoding.
let activePlanes = 0;
const planeWaiters: Array<() => void> = [];
function acquirePlaneSlot(signal: AbortSignal): Promise<() => void> {
  return new Promise((resolve, reject) => {
    const abort = () => {
      const index = planeWaiters.indexOf(take);
      if (index >= 0) planeWaiters.splice(index, 1);
      reject(signal.reason);
    };
    const take = () => {
      signal.removeEventListener('abort', abort);
      activePlanes++;
      let released = false;
      resolve(() => {
        if (released) return;
        released = true;
        activePlanes--;
        planeWaiters.shift()?.();
      });
    };
    if (signal.aborted) { reject(signal.reason); return; }
    if (activePlanes < 2) take();
    else {
      planeWaiters.push(take);
      signal.addEventListener('abort', abort, { once: true });
    }
  });
}
let rendering: Promise<unknown> = Promise.resolve();
async function renderPixels(
  file: QueuedFile & { result: NonNullable<QueuedFile['result']> },
  renderer: Renderer,
  format: ExportFormat,
  quality: number,
  signal: AbortSignal,
): Promise<EncodeJob> {
  const { exportData } = file.result;

  // Compute merged OpenDRT config (per-file preset + overrides)
  const baseConfig = configFromPreset(file.edit.lookPreset);
  const sdrConfig = configWithOverrides(
    baseConfig,
    file.edit.openDrtOverrides,
    file.edit.preProcessOverrides,
  );
  const sdrTs = computeTonescaleParams(sdrConfig);

  let data: Float32Array;
  let hdrData: Float32Array | null = null;
  let peakLuminance = sdrConfig.peak_luminance;

  if (exportFormatInfo(format).needsHdr) {
    // JPEG-HDR and AVIF need HDR tonemapped data
    const hdrConfig = deriveHdrConfig(sdrConfig, HDR_PEAK_LUMINANCE);
    const hdrTs = computeTonescaleParams(hdrConfig);
    peakLuminance = HDR_PEAK_LUMINANCE;

    if (format === 'jpeg-hdr') {
      // Dual render: SDR (Rec.709) + HDR (Rec.2020)
      data = await renderer.readback(sdrConfig, sdrTs, 'rec709');
      signal.throwIfAborted();
      hdrData = await renderer.readback(hdrConfig, hdrTs, 'rec2020');
    } else {
      // AVIF: HDR only (Rec.2020)
      data = await renderer.readback(hdrConfig, hdrTs, 'rec2020');
    }
  } else {
    // JPEG / TIFF: SDR (Rec.709)
    data = await renderer.readback(sdrConfig, sdrTs, 'rec709');
  }

  return {
    data,
    hdrData,
    width: exportData.width,
    height: exportData.height,
    orientation: exportData.orientation,
    format,
    quality,
    peakLuminance,
    signal,
  };
}
export function enqueueExport(
  fileId: string,
  format = useAppStore.getState().exportFormat,
  quality = useAppStore.getState().exportQuality,
  observer?: ExportObserver,
): ExportJobHandle {
  const host = getHost();
  const snapshot = useAppStore.getState();
  const file = snapshot.files.find((f) => f.id === fileId);
  const lease = acquireResult(fileId);
  const controller = new AbortController();
  let state: ExportState = 'queued';
  const setState = (next: ExportState) => {
    if (state === next) return;
    state = next;
    observer?.state?.(next);
  };
  let releaseSlot: (() => void) | undefined;
  let released = false;
  const release = () => {
    if (!released) {
      released = true;
      lease?.release();
    }
  };
  let cancelQueued!: (value: null) => void;
  const queuedCancellation = new Promise<null>((resolve) => {
    cancelQueued = resolve;
  });
  const work = (async () => {
    try {
      if (!file?.result || !lease) throw new Error('nothing to export');
      if (file.processedKey !== null && file.processedKey !== processingKey(file, snapshot)) {
        throw new Error('Wait for this photo to finish processing before exporting.');
      }
      const availability = await host.exporter.status();
      if (!availability.available) throw new Error(availability.reason);
      controller.signal.throwIfAborted();
      const destination = await host.exporter.chooseDestination(
        file.id,
        `${file.name}.${exportFormatInfo(format).ext}`,
        format,
      );
      if (!destination) {
        setState('cancelled');
        return null;
      }
      controller.signal.throwIfAborted();
      observer?.destination?.(destination);
      releaseSlot = await acquirePlaneSlot(controller.signal);
      const readback = rendering
        .catch(() => {})
        .then(async () => {
          controller.signal.throwIfAborted();
          setState('rendering');
          // A separate renderer owns the pinned texture throughout both readbacks, independent of selection.
          const renderer = await createRenderer(document.createElement('canvas'));
          try {
            renderer.setImage(lease.image.gpu);
            return await renderPixels(
              file as QueuedFile & { result: NonNullable<QueuedFile['result']> },
              renderer,
              format,
              quality,
              controller.signal,
            );
          } finally {
            renderer.dispose();
            release();
          }
        });
      rendering = readback;
      const pixels = await readback;
      controller.signal.throwIfAborted();
      setState('encoding');
      const result = await host.exporter.encode(pixels, destination);
      controller.signal.throwIfAborted();
      setState('done');
      return result;
    } catch (error) {
      if (controller.signal.aborted) {
        setState('cancelled');
        return null;
      }
      setState('failed');
      throw error;
    } finally {
      releaseSlot?.();
      release();
    }
  })();
  return {
    get state() {
      return state;
    },
    promise: Promise.race([work, queuedCancellation]),
    cancel: () => {
      controller.abort();
      if (state === 'queued') {
        setState('cancelled');
        release();
        cancelQueued(null);
      }
    },
  };
}
/** Golden uses the same host encoder, with delivery disabled by the web entry. */
export async function renderExport(
  fileId: string,
  format = useAppStore.getState().exportFormat,
  quality = useAppStore.getState().exportQuality,
): Promise<{ blob: Blob; ext: string }> {
  const result = await enqueueExport(fileId, format, quality).promise;
  if (!result?.blob) throw new Error('The exporter returned no bytes for the golden check.');
  return { blob: result.blob, ext: exportFormatInfo(format).ext };
}
