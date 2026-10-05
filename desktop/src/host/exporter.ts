import type { ExportHost } from '@/host';
import type { DesktopBridge } from '../protocol/bridge';
import { EXPORT_CHUNK_BYTES } from '../protocol/rpc';
import type { WorkerClient } from './port';
const LONG_MS = 10 * 60_000;
export function createExporter(bridge: DesktopBridge, client: WorkerClient, opts: { chunkBytes?: number; platform?: string } = {}): ExportHost {
  const chunkBytes = opts.chunkBytes ?? EXPORT_CHUNK_BYTES;
  if (!Number.isSafeInteger(chunkBytes) || chunkBytes <= 0 || chunkBytes % 4 !== 0 || chunkBytes > EXPORT_CHUNK_BYTES) throw new Error('Invalid export chunk size');
  const platform = opts.platform ?? navigator.platform;
  return {
    async status() {
      try { return (await client.request({ op: 'exportStatus' })).availability ?? { available: false, reason: 'The worker did not report export status.' }; }
      catch (error) { return { available: false, reason: error instanceof Error ? error.message : String(error) }; }
    },
    async chooseDestination(photoId, _suggestedName, format) {
      const choice = await bridge.chooseExportDestination(photoId, format);
      return choice ? { token: choice.token } : null;
    },
    async encode(job, destination) {
      const id = crypto.randomUUID(), generation = client.generation;
      const bound = { generation, signal: job.signal };
      let cancelled = false;
      const cancel = () => {
        if (cancelled) return;
        cancelled = true;
        void client.request({ op: 'exportCancel', job: id }, { generation }).catch(() => {});
      };
      const check = () => { if (job.signal?.aborted) throw new DOMException('Export cancelled', 'AbortError'); };
      check();
      job.signal?.addEventListener('abort', cancel, { once: true });
      try {
        const count = job.hdrData ? 2 : 1;
        await client.request({ op: 'exportBegin', job: id, destination: destination.token, format: job.format, width: job.width,
          height: job.height, orientation: job.orientation, quality: Math.round(job.quality), peakLuminance: job.peakLuminance,
          planes: count as 1 | 2 }, { ...bound, timeoutMs: LONG_MS });
        for (let index = 0; index < count; index++) {
          // Scope the borrowed plane to its transfer; no planes array retains sent data.
          let plane: Float32Array | null = index === 0 ? job.data : job.hdrData;
          if (!plane) throw new Error('Missing export plane');
          for (let offset = 0; offset < plane.byteLength; offset += chunkBytes) {
            check();
            const end = Math.min(offset + chunkBytes, plane.byteLength);
            const data = plane.buffer.slice(plane.byteOffset + offset, plane.byteOffset + end) as ArrayBuffer;
            await client.request({ op: 'exportChunk', job: id, plane: index as 0 | 1, offset, data }, bound);
          }
          plane = null;
          if (index === 0) job.data = new Float32Array(0); else job.hdrData = null;
        }
        check();
        const { receipt } = await client.request({ op: 'exportCommit', job: id }, { ...bound, timeoutMs: LONG_MS });
        check();
        if (!receipt) throw new Error('The worker did not confirm the export.');
        return { bytes: receipt.bytes, sha256: receipt.sha256, encodeMs: receipt.encodeMs, name: receipt.name };
      } catch (error) { cancel(); check(); throw error; }
      finally { job.signal?.removeEventListener('abort', cancel); }
    },
    reveal: { label: /^Win/i.test(platform) ? 'Show in Explorer' : 'Show in Finder', open: async d => bridge.revealExport(d.token) },
  };
}
