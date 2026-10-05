import { afterEach, expect, it, vi } from 'vitest';
import type { DesktopBridge } from '../protocol/bridge';
import type { EncodeJob } from '@/host';
import type { PortRequest } from '../protocol/rpc';
import { createWorkerClient } from './port';
import { createExporter } from './exporter';
const receipt = { bytes: 123, sha256: 'a'.repeat(64), encodeMs: 12, name: 'x.avif' };
const tick = async () => { for (let i = 0; i < 200; i++) await Promise.resolve(); };
function setup() {
 const win = new EventTarget(); vi.stubGlobal('window', win); vi.stubGlobal('location', { origin: 'app://bundle' });
 const ports: FakePort[] = [];
 class FakePort extends EventTarget {
  sent: PortRequest[] = []; hold: string | undefined; hook?: (r: PortRequest) => void;
  start() {} close() {} receive(r: PortRequest) { this.dispatchEvent(new MessageEvent('message', { data: { v: 1, rid: r.rid, ok: true, ...(r.op === 'exportCommit' ? { receipt } : {}), ...(r.op === 'exportStatus' ? { availability: { available: true } } : {}) } })); }
  postMessage(r: PortRequest, transfer?: unknown) { expect(transfer).toBeUndefined(); this.sent.push(r); this.hook?.(r); if (this.hold !== r.op) queueMicrotask(() => this.receive(r)); }
 }
 const bridge = { chooseExportDestination: vi.fn(async () => ({ token: 'destination' })), revealExport: vi.fn(async () => {}), requestWorkerPort: vi.fn(async requestId => { const port = new FakePort(); ports.push(port); const e = new MessageEvent('message', { data: { type: 'xveon-port', version: 2, requestId }, origin: 'app://bundle' }); Object.defineProperties(e, { source: { value: win }, ports: { value: [port] } }); win.dispatchEvent(e); }) } as unknown as DesktopBridge;
 const client = createWorkerClient(bridge, () => {});
 return { bridge, client, ports, exporter: createExporter(bridge, client, { chunkBytes: 1024, platform: 'MacIntel' }) };
}
const job = (): EncodeJob => ({ format: 'jpeg-hdr', data: new Float32Array(3000), hdrData: new Float32Array(3000), width: 1000, height: 1, orientation: '1', quality: 80.4, peakLuminance: 1000 });
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });
it('returns availability, destination and reveal capabilities over one connection', async () => {
 const { bridge, client, exporter } = setup(); expect(await exporter.status()).toEqual({ available: true }); expect(await exporter.chooseDestination('a'.repeat(22), 'x.avif', 'avif')).toEqual({ token: 'destination' }); expect(bridge.chooseExportDestination).toHaveBeenCalledWith('a'.repeat(22), 'avif');
 vi.mocked(bridge.chooseExportDestination).mockResolvedValueOnce(null); expect(await exporter.chooseDestination('a'.repeat(22), 'x.avif', 'avif')).toBeNull();
 expect(exporter.reveal!.label).toBe('Show in Finder'); await exporter.reveal!.open({ token: 'destination' }); expect(bridge.revealExport).toHaveBeenCalledWith('destination'); expect(createExporter(bridge, client, { platform: 'Win32' }).reveal!.label).toBe('Show in Explorer'); expect(bridge.requestWorkerPort).toHaveBeenCalledTimes(1);
 client.stop('Worker stopped'); expect(await exporter.status()).toEqual({ available: false, reason: 'Worker stopped' });
});
it('chunks both planes exactly and releases each after its final acknowledgment', async () => {
 const { exporter, ports } = setup(); const j = job(); ports[0].hook = r => { if (r.op === 'exportChunk' && r.plane === 1) expect(j.data.length).toBe(0); if (r.op === 'exportCommit') expect(j.hdrData).toBeNull(); };
 expect(await exporter.encode(j, { token: 'destination' })).toEqual(receipt); const sent = ports[0].sent; expect(sent[0]).toMatchObject({ op: 'exportBegin', planes: 2, quality: 80 }); expect(sent.at(-1)).toMatchObject({ op: 'exportCommit' });
 for (const plane of [0, 1]) { const chunks = sent.filter(r => r.op === 'exportChunk' && r.plane === plane); expect(chunks.map(r => r.op === 'exportChunk' && [r.offset, r.data.byteLength])).toEqual(Array.from({ length: 12 }, (_, i) => [i * 1024, i === 11 ? 736 : 1024])); for (const r of chunks) if (r.op === 'exportChunk') expect(r.data).toBeInstanceOf(ArrayBuffer); }
 expect(j.data.length).toBe(0); expect(j.hdrData).toBeNull();
});
it('sends only the bytes of a subarray without detaching it', async () => {
 const { exporter, ports } = setup(); const source = new Float32Array([9, 1, 2, 3, 9]); const j = { ...job(), format: 'avif' as const, data: source.subarray(1, 4), hdrData: null, width: 1 }; await exporter.encode(j, { token: 'destination' }); const r = ports[0].sent.find(r => r.op === 'exportChunk')!; if (r.op === 'exportChunk') expect([...new Float32Array(r.data)]).toEqual([1, 2, 3]); expect(source.byteLength).toBe(20);
});
it.each(['exportBegin', 'exportChunk', 'exportCommit'])('aborts while %s is pending', async op => {
 const { exporter, ports, client } = setup(); const abort = new AbortController(); ports[0].hold = op; const pending = exporter.encode({ ...job(), signal: abort.signal }, { token: 'destination' }); const failed = expect(pending).rejects.toMatchObject({ name: 'AbortError' }); await tick(); expect(ports[0].sent.at(-1)?.op).toBe(op); abort.abort(); await failed; await tick(); expect(ports[0].sent.filter(r => r.op === 'exportCancel')).toHaveLength(1); const count = ports[0].sent.length; ports[0].receive(ports[0].sent.find(r => r.op === op)!); await tick(); expect(ports[0].sent).toHaveLength(count); client.stop('cleanup');
});
it.each(['pending', 'between'])('pins the transfer and cancellation to its generation (%s)', async mode => {
 const { exporter, ports, client } = setup(); if (mode === 'pending') ports[0].hold = 'exportChunk'; else ports[0].hook = r => { if (r.op === 'exportChunk') { ports[0].receive(r); void client.restart(); } };
 const pending = exporter.encode(job(), { token: 'destination' }); const failed = expect(pending).rejects.toThrow('Worker restarted'); await tick(); if (mode === 'pending') await client.restart(); await failed; await tick(); expect(ports[1].sent).toEqual([]); client.stop('cleanup');
});
