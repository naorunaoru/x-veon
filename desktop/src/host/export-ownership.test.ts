// @vitest-environment jsdom
import { setFlagsFromString } from 'node:v8';
import { runInNewContext } from 'node:vm';
import { beforeEach, expect, it, vi } from 'vitest';
import { fakeHost, fakePhoto } from '@/test/fake-host';
import { fromLibraryPhoto } from '@/app/store/photo';
import { useAppStore } from '@/app/store';
import { setHost } from '@/app/services/host';
import { enqueueExport } from '@/app/services/export';
import { createExporter } from './exporter';
import type { DesktopBridge } from '../protocol/bridge';
import type { WorkerClient } from './port';
const mocks = vi.hoisted(() => ({ readback: vi.fn() }));
vi.mock('@/renderer', () => ({ createRenderer: async () => ({ readback: mocks.readback, setImage() {}, dispose() {} }) }));
vi.mock('@/app/services/processing', () => ({ acquireResult: () => ({ image: { gpu: {} }, release() {} }) }));
setFlagsFromString('--expose-gc');
const gc = runInNewContext('gc') as () => void;
beforeEach(() => {
 const file = fromLibraryPhoto(fakePhoto());
 file.result = { exportData: { width: 1, height: 1, orientation: '', xyzToCam: null, wbCoeffs: new Float32Array(3), camToXyz: new Float32Array(12) }, metadata: {} as never };
 useAppStore.setState({ files: [file] }); mocks.readback.mockReset();
});
it.each(['begin-denied', 'cancelled'] as const)('releases both rendered planes after desktop %s while the queue is idle, then progresses', async ending => {
 const refs: WeakRef<Float32Array>[] = [];
 mocks.readback.mockImplementation(async () => { const plane = new Float32Array([1, 2, 3]); refs.push(new WeakRef(plane)); return plane; });
 let cancel!: () => void;
 const client = { generation: 1, request: async (message: { op: string }) => {
  if (message.op === 'exportStatus') return { availability: { available: true } };
  if (message.op === 'exportBegin' && ending === 'begin-denied') throw new Error('Begin denied');
  if (message.op === 'exportChunk') cancel();
  return {};
 } } as unknown as WorkerClient;
 const host = fakeHost();
 host.exporter = createExporter({ chooseExportDestination: async () => ({ token: 'destination' }) } as unknown as DesktopBridge, client, { platform: 'MacIntel', chunkBytes: 4 });
 setHost(host);
 const job = enqueueExport('a', 'jpeg-hdr'); cancel = job.cancel;
 if (ending === 'begin-denied') await expect(job.promise).rejects.toThrow('Begin denied'); else await expect(job.promise).resolves.toBeNull();
 expect(job.state).toBe(ending === 'begin-denied' ? 'failed' : 'cancelled'); expect(refs).toHaveLength(2);
 // Mock call/result history would itself own the readback promises. Remove it
 // before collecting, and never dereference within the collection turns.
 mocks.readback.mockClear();
 for (let i = 0; i < 5; i++) { await new Promise(resolve => setTimeout(resolve, 0)); gc(); }
 expect(refs.map(ref => ref.deref())).toEqual([undefined, undefined]);
 setHost(fakeHost()); await expect(enqueueExport('a', 'tiff').promise).resolves.not.toBeNull();
});
it('keeps a readback failure on its own job and allows the next render', async () => {
 setHost(fakeHost()); mocks.readback.mockRejectedValueOnce(new Error('GPU readback failed')).mockResolvedValue(new Float32Array(3));
 await expect(enqueueExport('a', 'tiff').promise).rejects.toThrow('GPU readback failed');
 await expect(enqueueExport('a', 'tiff').promise).resolves.not.toBeNull();
});
