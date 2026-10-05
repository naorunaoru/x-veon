import { EventEmitter } from 'node:events';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { createWorkerSupervisor } from './worker';
import { createWorkerController } from '../worker/controller';
import { createExportService } from '../worker/exports';
import type { ExportBegin } from '../protocol/rpc';
import type { NativeEncoder } from '../worker/native';
let dir: string;
beforeEach(async () => { dir = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'xveon-stop-'))); });
afterEach(async () => { vi.restoreAllMocks(); vi.useRealTimers(); await fs.rm(dir, { recursive: true, force: true }); });
const deferred = () => { let resolve!: () => void; const promise = new Promise<void>(r => { resolve = r; }); return { promise, resolve }; };
function harness(encode: NativeEncoder['encode'] = async () => new Uint8Array([1, 2, 3]), stopTimeoutMs = 3000) {
 const exports = createExportService({ native: () => ({ ok: true, encoder: { encode } }) });
 const child = Object.assign(new EventEmitter(), { pid: 42, kill: vi.fn(() => { child.emit('exit'); return true; }), postMessage: (data: unknown) => { void controller.handleMain(data); } });
 const controller = createWorkerController({ exports, postToMain: data => child.emit('message', data) });
 const fork = vi.fn(() => child), onEvent = vi.fn();
 const supervisor = createWorkerSupervisor({ fork, sessionKey: Buffer.from('key'), cacheDir: dir, onEvent, stopTimeoutMs });
 async function ready(job: string) {
  await supervisor.request({ kind: 'export-destination', token: job, path: path.join(dir, job + '.avif') });
  const message: ExportBegin = { v: 1, rid: 1, op: 'exportBegin', job, destination: job, format: 'avif', width: 1, height: 1, orientation: '', quality: 95, peakLuminance: 1000, planes: 1 };
  await exports.begin(message); exports.chunk(job, 0, 0, new Float32Array([1, 2, 3]).buffer);
 }
 return { exports, child, supervisor, ready, fork, onEvent };
}
it('main stop waits for the real worker atomic temp cleanup before killing and cancels queued exports', async () => {
 const h = harness(), gate = deferred(), written = deferred();
 await h.ready('job-0001'); await h.ready('job-0002');
 await fs.writeFile(path.join(dir, 'job-0001.avif'), 'original');
 const write = fs.writeFile.bind(fs);
 vi.spyOn(fs, 'writeFile').mockImplementationOnce(async (...args) => { await write(...args); written.resolve(); await gate.promise; });
 const first = h.exports.commit('job-0001').catch(error => error.message);
 const second = h.exports.commit('job-0002').catch(error => error.message);
 await written.promise;
 expect((await fs.readdir(dir)).some(name => name.endsWith('.tmp'))).toBe(true);
 const stopped = h.supervisor.stop();
 expect(h.supervisor.stop()).toBe(stopped);
 expect(h.child.kill).not.toHaveBeenCalled();
 await expect(h.supervisor.ready()).rejects.toThrow('stopped');
 await expect(h.supervisor.restart()).rejects.toThrow('stopped');
 gate.resolve(); await stopped;
 expect(await first).toBe('Export cancelled'); expect(await second).toBe('Export cancelled');
 expect(await fs.readdir(dir)).toEqual(['job-0001.avif']);
 expect(await fs.readFile(path.join(dir, 'job-0001.avif'), 'utf8')).toBe('original');
 expect(h.child.kill).toHaveBeenCalledOnce(); expect(h.fork).toHaveBeenCalledOnce(); expect(h.onEvent).not.toHaveBeenCalled();
});
it('ACKs shutdown without waiting for a never-resolving native encode or mutating its borrowed planes', async () => {
 let borrowed!: Float32Array;
 const started = deferred(); const h = harness(data => { borrowed = data; started.resolve(); return new Promise(() => {}); });
 await h.ready('job-0001'); void h.exports.commit('job-0001'); await started.promise;
 await h.supervisor.stop();
 expect([...borrowed]).toEqual([1, 2, 3]); expect(h.child.kill).toHaveBeenCalledOnce(); expect(await fs.readdir(dir)).toEqual([]);
});
it('uses a finite fallback for an unresponsive worker and ignores unrelated ACKs', async () => {
 vi.useFakeTimers(); const h = harness(undefined, 50); await h.supervisor.ready();
 const send = vi.spyOn(h.child, 'postMessage').mockImplementation(() => {});
 const stopped = h.supervisor.stop(); const message = send.mock.calls[0][0] as { rid: number };
 h.child.emit('message', { v: 1, kind: 'shutdown', rid: message.rid + 1 });
 await vi.advanceTimersByTimeAsync(49); expect(h.child.kill).not.toHaveBeenCalled();
 await vi.advanceTimersByTimeAsync(1); await stopped; expect(h.child.kill).toHaveBeenCalledOnce(); expect(h.fork).toHaveBeenCalledOnce();
});
