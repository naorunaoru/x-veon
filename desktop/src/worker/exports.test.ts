import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { createHash } from 'node:crypto';
import { beforeEach, afterEach, it, expect, vi } from 'vitest';
import { requireFileSymlinks, directoryLinkType } from '../test/symlinks';
import { createExportService } from './exports';
import type { ExportBegin } from '../protocol/rpc';
import type { NativeEncoder } from './native';
const deferred = () => { let resolve!: () => void; const promise = new Promise<void>(r => { resolve = r; }); return { promise, resolve }; };
const message = (job = 'job-0001', destination = job): ExportBegin => ({ v: 1, rid: 1, op: 'exportBegin', job, destination, format: 'avif', width: 1, height: 1, orientation: '', quality: 95, peakLuminance: 1000, planes: 1 });
let dir: string;
beforeEach(async () => { dir = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'xveon-export-'))); });
afterEach(async () => { vi.restoreAllMocks(); await fs.rm(dir, { recursive: true, force: true }); });
function setup(encode: NativeEncoder['encode'] = async (d, _h, i) => new Uint8Array([i.width, i.height, d.length % 256]), slots = 2) {
 const encoder = { encode: vi.fn(encode) }; const service = createExportService({ native: () => ({ ok: true, encoder }), slots });
 async function begin(id: string, name = id + '.avif') { service.register(id, path.join(dir, name)); await service.begin(message(id)); }
 async function ready(id: string, name?: string) { await begin(id, name); service.chunk(id, 0, 0, new Float32Array([1, 2, 3]).buffer); }
 return { service, encoder, begin, ready };
}
it.each(['test.avif', 'Тест – 写真.avif'])('writes the registered %s and returns its bytes and hash', async name => {
 const h = setup(); await h.ready('job-0001', name); const receipt = await h.service.commit('job-0001'); const data = await fs.readFile(path.join(dir, name));
 expect(data).toEqual(Buffer.from([1, 1, 3])); expect(receipt).toEqual({ name, bytes: 3, sha256: createHash('sha256').update(data).digest('hex'), encodeMs: expect.any(Number) }); expect(receipt.encodeMs).toBeGreaterThanOrEqual(0);
});
it('encodes commits serially in commit order', async () => {
 const gate = deferred(); let active = 0, max = 0; const order: number[] = [];
 const h = setup(async data => { active++; max = Math.max(max, active); order.push(data[0]); await gate.promise; active--; return new Uint8Array([1]); });
 await h.begin('job-0001'); await h.begin('job-0002'); h.service.chunk('job-0001', 0, 0, new Float32Array([1, 0, 0]).buffer); h.service.chunk('job-0002', 0, 0, new Float32Array([2, 0, 0]).buffer);
 const second = h.service.commit('job-0002'), first = h.service.commit('job-0001'); await vi.waitFor(() => expect(order).toEqual([2])); gate.resolve(); await Promise.all([first, second]); expect(order).toEqual([2, 1]); expect(max).toBe(1);
});
it('consumes each token once and rejects unregistered tokens', async () => {
 const h = setup(); await h.begin('job-0001'); await expect(h.service.begin(message('job-0002', 'job-0001'))).rejects.toThrow('Unknown or used export destination'); await expect(h.service.begin(message('job-0003'))).rejects.toThrow('Unknown or used export destination'); h.service.cancelAll();
});
it('rejects out-of-order, overflow, invalid planes, incomplete and duplicate commits', async () => {
 const gate = deferred(); const h = setup(async () => { await gate.promise; return new Uint8Array([1]); }); await h.begin('job-0001');
 expect(() => h.service.chunk('job-0001', 0, 4, new ArrayBuffer(4))).toThrow('Export data out of order');
 expect(() => h.service.chunk('job-0001', 0, 0, new ArrayBuffer(16))).toThrow('overflows'); expect(() => h.service.chunk('job-0001', 1, 0, new ArrayBuffer(4))).toThrow('Invalid export plane');
 await expect(h.service.commit('job-0001')).rejects.toThrow('Incomplete export data'); h.service.chunk('job-0001', 0, 0, new ArrayBuffer(12));
 const pending = h.service.commit('job-0001'); await expect(h.service.commit('job-0001')).rejects.toThrow('Export already committed'); expect(() => h.service.chunk('job-0001', 0, 0, new ArrayBuffer(4))).toThrow('Export data after commit'); gate.resolve(); await pending; expect(h.encoder.encode).toHaveBeenCalledOnce();
});
it.each(['queued', 'encoding'] as const)('cancels %s without changing an existing destination', async state => {
 const gate = deferred(); const h = setup(async data => { await gate.promise; expect([...data]).toEqual([1, 2, 3]); return new Uint8Array([1]); });
 await h.ready('job-0001'); await h.ready('job-0002'); await fs.writeFile(path.join(dir, 'job-0002.avif'), 'original'); await fs.writeFile(path.join(dir, 'job-0001.avif'), 'original');
 const first = h.service.commit('job-0001'); const second = h.service.commit('job-0002'); const outcome = expect(state === 'queued' ? second : first).rejects.toThrow('Export cancelled');
 await vi.waitFor(() => expect(h.encoder.encode).toHaveBeenCalledOnce()); h.service.cancel(state === 'queued' ? 'job-0002' : 'job-0001'); gate.resolve(); await outcome;
 if (state === 'queued') { await first; expect(h.encoder.encode).toHaveBeenCalledOnce(); expect(await fs.readFile(path.join(dir, 'job-0002.avif'), 'utf8')).toBe('original'); }
 else { await second; expect(await fs.readFile(path.join(dir, 'job-0001.avif'), 'utf8')).toBe('original'); }
});
it('waits for slots, cancels waiting begins, and releases slots exactly once', async () => {
 const gate = deferred(); const h = setup(async () => { await gate.promise; return new Uint8Array([1]); }); await h.ready('job-0001'); await h.begin('job-0002');
 let done = false; const third = h.begin('job-0003').then(() => { done = true; }); await new Promise(r => setTimeout(r, 10)); expect(done).toBe(false);
 await expect(h.service.begin(message('job-0003'))).rejects.toThrow('Duplicate export job'); const cancelled = expect(third).rejects.toThrow('Export cancelled'); h.service.cancel('job-0003'); await cancelled;
 const fourth = h.begin('job-0004').then(() => { done = true; }); const first = h.service.commit('job-0001'); await vi.waitFor(() => expect(h.encoder.encode).toHaveBeenCalledOnce()); expect(done).toBe(false); gate.resolve(); await first; await fourth;
 let fifthDone = false; const fifth = h.begin('job-0005').then(() => { fifthDone = true; }); await new Promise(r => setTimeout(r, 10)); expect(fifthDone).toBe(false); h.service.cancel('job-0002'); await fifth; h.service.cancelAll();
});
it('reserves pending IDs and cancels directory lookups without reviving jobs', async () => {
 const gate = deferred(); const realpath = fs.realpath.bind(fs); vi.spyOn(fs, 'realpath').mockImplementationOnce(async (...args) => { await gate.promise; return realpath(...args); });
 const h = setup(); const begin = h.begin('job-0001'); await expect(h.service.begin(message())).rejects.toThrow('Duplicate export job'); const outcome = expect(begin).rejects.toThrow('Export cancelled'); h.service.cancelAll(); gate.resolve(); await outcome;
 expect(() => h.service.chunk('job-0001', 0, 0, new ArrayBuffer(12))).toThrow('Unknown export job'); await h.begin('job-0002'); await h.begin('job-0003'); h.service.cancelAll();
});
it('unknown cancellation does not poison future IDs; cancelAll drops receiving and waiting jobs', async () => {
 const h = setup(); h.service.cancel('job-0001'); await h.begin('job-0001'); await h.begin('job-0002'); const third = h.begin('job-0003'); const outcome = expect(third).rejects.toThrow('Export cancelled'); h.service.cancelAll(); await outcome;
 for (const id of ['job-0001', 'job-0002', 'job-0003']) expect(() => h.service.chunk(id, 0, 0, new ArrayBuffer(12))).toThrow('Unknown export job'); await h.begin('job-0004'); await h.begin('job-0005'); h.service.cancelAll();
});
it('reports unavailable native encoding', async () => {
 const service = createExportService({ native: () => ({ ok: false, reason: 'no addon' }) }); expect(service.status()).toEqual({ available: false, reason: 'no addon' }); await expect(service.begin(message())).rejects.toThrow('no addon');
});
it('preserves OS error code and cause when the destination folder disappears', async () => {
 const h = setup(); await h.ready('job-0001'); await fs.rm(dir, { recursive: true }); await expect(h.service.commit('job-0001')).rejects.toMatchObject({ message: expect.stringMatching(/Couldn't write job-0001.avif:.*ENOENT/), code: 'ENOENT', cause: expect.objectContaining({ code: 'ENOENT' }) });
});
it('rejects a symlink replacement of the folder', async context => {
 requireFileSymlinks(context); const h = setup(); await h.ready('job-0001'); const moved = dir + '-moved'; await fs.rename(dir, moved); await fs.symlink(moved, dir, directoryLinkType);
 try { await expect(h.service.commit('job-0001')).rejects.toThrow('The folder changed while saving'); expect(await fs.readdir(moved)).toEqual([]); } finally { await fs.unlink(dir); await fs.rename(moved, dir); }
});
it('atomically replaces an existing file without leaving a temp', async () => { const h = setup(); await h.ready('job-0001'); await fs.writeFile(path.join(dir, 'job-0001.avif'), 'old'); await h.service.commit('job-0001'); expect(await fs.readdir(dir)).toEqual(['job-0001.avif']); expect(await fs.readFile(path.join(dir, 'job-0001.avif'))).toEqual(Buffer.from([1, 1, 3])); });
it('cancels during the write, preserving the destination and cleaning its temp', async () => {
 const h = setup(); await h.ready('job-0001'); const target = path.join(dir, 'job-0001.avif'); await fs.writeFile(target, 'original'); const write = fs.writeFile.bind(fs);
 vi.spyOn(fs, 'writeFile').mockImplementationOnce(async (...args) => { await write(...args); h.service.cancel('job-0001'); });
 await expect(h.service.commit('job-0001')).rejects.toThrow('Export cancelled'); expect(await fs.readFile(target, 'utf8')).toBe('original'); expect(await fs.readdir(dir)).toEqual(['job-0001.avif']);
});
it('cancelAll cancels queued and encoding work without damaging borrowed planes', async () => {
 const gate = deferred(); const h = setup(async data => { await gate.promise; expect([...data]).toEqual([1, 2, 3]); return new Uint8Array([1]); });
 await h.ready('job-0001'); await h.ready('job-0002');
 const first = h.service.commit('job-0001'), second = h.service.commit('job-0002'); const outcomes = [expect(first).rejects.toThrow('Export cancelled'), expect(second).rejects.toThrow('Export cancelled')];
 await vi.waitFor(() => expect(h.encoder.encode).toHaveBeenCalledOnce()); h.service.cancelAll(); gate.resolve(); await Promise.all(outcomes);
 expect(h.encoder.encode).toHaveBeenCalledOnce(); expect(await fs.readdir(dir)).toEqual([]);
 for (const id of ['job-0001', 'job-0002']) expect(() => h.service.chunk(id, 0, 0, new ArrayBuffer(12))).toThrow('Unknown export job');
 await h.begin('job-0003'); await h.begin('job-0004'); h.service.cancelAll();
});
it('recovers the queue and slots after an encoder rejects', async () => {
 const h = setup(); h.encoder.encode.mockRejectedValueOnce(new Error('encoder failed')); await h.ready('job-0001'); await h.ready('job-0002');
 const first = h.service.commit('job-0001'), second = h.service.commit('job-0002'); await expect(first).rejects.toThrow('encoder failed'); await second; await h.begin('job-0003'); await h.begin('job-0004'); h.service.cancelAll();
});
it('reports write failures using the destination name and preserving the original cause', async () => {
 const h = setup(); await h.ready('job-0001'); let original: Error;
 vi.spyOn(fs, 'open').mockImplementationOnce(async file => { original = Object.assign(new Error(`ENOSPC: no space left on device, open '${file}'`), { code: 'ENOSPC' }); throw original; });
 const error = await h.service.commit('job-0001').catch(e => e); expect(error.code).toBe('ENOSPC'); expect(error.cause).toBe(original!); expect(error.message).toContain("Couldn't write job-0001.avif: ENOSPC"); expect(error.message).not.toContain('.tmp');
});
it('collects both HDR planes in ordered chunks before encoding', async () => {
 const h = setup(); h.service.register('dest-001', path.join(dir, 'hdr.jpg'));
 await h.service.begin({ ...message('job-0001', 'dest-001'), format: 'jpeg-hdr', planes: 2 });
 h.service.chunk('job-0001', 0, 0, new Float32Array([1]).buffer); h.service.chunk('job-0001', 0, 4, new Float32Array([2, 3]).buffer);
 await expect(h.service.commit('job-0001')).rejects.toThrow('Incomplete export data');
 h.service.chunk('job-0001', 1, 0, new Float32Array([4, 5, 6]).buffer); await h.service.commit('job-0001');
 expect(h.encoder.encode).toHaveBeenCalledWith(new Float32Array([1, 2, 3]), new Float32Array([4, 5, 6]), expect.objectContaining({ format: 'jpeg-hdr' }));
});
