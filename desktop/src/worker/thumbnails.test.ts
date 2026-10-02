import fs from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { beforeEach, afterEach, it, expect, vi } from 'vitest';
import { createCache } from './cache';
import { listFolder } from './folder';
import { ensureHead } from './thumbnails';
let dir: string;
beforeEach(async () => { dir = await fs.mkdtemp(path.join(os.tmpdir(), 'xveon-head-')); });
afterEach(async () => { vi.restoreAllMocks(); await fs.rm(dir, { recursive: true, force: true }); });
function raf() {
 const bytes = Buffer.alloc(0x80); bytes.write('FUJIFILMCCD-RAW'); bytes.write('X-T5', 0x1c); bytes.writeUInt32BE(0x70, 0x54); bytes.writeUInt32BE(4, 0x58); bytes.set([255, 216, 255, 217], 0x70); return bytes;
}
it('caches JPEG bytes and quick metadata; a second call never opens the RAW', async () => {
 await fs.writeFile(path.join(dir, 'a.RAF'), raf()); const entry = (await listFolder(dir))[0]; const cache = createCache({ dir: path.join(dir, 'cache') });
 const result = await ensureHead(entry, cache);
 expect(result.metadata).toEqual({ camera: 'Fujifilm X-T5', lensModel: '', focalLength: 0, fNumber: 0 });
 expect(result.thumbnail).not.toBeNull(); expect(await fs.readFile(result.thumbnail!)).toEqual(Buffer.from([255, 216, 255, 217]));
 const open = vi.spyOn(fs, 'open'); expect(await ensureHead(entry, createCache({ dir: path.join(dir, 'cache') }))).toEqual(result); expect(open).not.toHaveBeenCalled();
});
it('caches negative extraction, then reads again after a file change or cache deletion', async () => {
 const raw = path.join(dir, 'a.ARW'); await fs.writeFile(raw, 'not a RAF or TIFF'); const entry = (await listFolder(dir))[0]; const cache = createCache({ dir: path.join(dir, 'cache') });
 expect(await ensureHead(entry, cache)).toEqual({ thumbnail: null, metadata: null });
 const open = vi.spyOn(fs, 'open'); expect(await ensureHead(entry, cache)).toEqual({ thumbnail: null, metadata: null }); expect(open).not.toHaveBeenCalled();
 await fs.writeFile(raw, raf()); const changed = (await listFolder(dir))[0]; expect((await ensureHead(changed, cache)).metadata?.camera).toBe('Fujifilm X-T5'); expect(open).toHaveBeenCalledTimes(1);
 await fs.rm(path.join(dir, 'cache'), { recursive: true }); await ensureHead(changed, cache); expect(open).toHaveBeenCalledTimes(2);
});
it('deduplicates concurrent requests for the same head', async () => {
 await fs.writeFile(path.join(dir, 'a.RAF'), raf()); const entry = (await listFolder(dir))[0]; const cache = createCache({ dir: path.join(dir, 'cache') }); const open = vi.spyOn(fs, 'open');
 const results = await Promise.all([ensureHead(entry, cache), ensureHead(entry, cache)]); expect(results[0]).toEqual(results[1]); expect(open).toHaveBeenCalledTimes(1);
});
const actual = path.resolve(__dirname, '../../../shared/public/samples/DSCF3332.RAF');
it('extracts the actual DSCF3332 RAF fixture with a bounded head read', async context => {
 if (!existsSync(actual)) context.skip('DSCF3332.RAF fixture absent from shared/public/samples');
 const stat = await fs.stat(actual); const entry = { path: actual, name: 'DSCF3332.RAF', size: stat.size, mtimeMs: stat.mtimeMs }; const cache = createCache({ dir });
 const openReal = fs.open.bind(fs); let readLength = 0;
 vi.spyOn(fs, 'open').mockImplementation(async (...args) => { const handle = await openReal(...args); const read = handle.read.bind(handle); vi.spyOn(handle, 'read').mockImplementation((...readArgs: any[]) => { readLength = readArgs[2]; return (read as any)(...readArgs); }); return handle; });
 const result = await ensureHead(entry, cache); expect(result.metadata?.camera).toContain('Fujifilm'); expect(result.thumbnail).not.toBeNull(); expect((await fs.readFile(result.thumbnail!)).subarray(0, 2)).toEqual(Buffer.from([255, 216])); expect(readLength).toBe(16 * 1024 ** 2);
});

it('holds at most four first-time heads through cache writes and releases slots after errors', async () => {
 const entries = Array.from({ length: 12 }, (_, i) => ({ path: path.join(dir, `${i}.RAF`), name: `${i}.RAF`, size: 128, mtimeMs: 1 }));
 await Promise.all(entries.map(e => fs.writeFile(e.path, raf())));
 const cache = createCache({ dir: path.join(dir, 'cache') });
 const releases: (() => void)[] = [];
 vi.spyOn(cache, 'putHeadMetadata').mockImplementation(async () => { await new Promise<void>(r => releases.push(r)); throw new Error('cache full'); });
 const opened = vi.spyOn(fs, 'open');
 const requests = entries.map(e => ensureHead(e, cache).catch(e => e.message));
 await vi.waitFor(() => expect(releases.length).toBeGreaterThanOrEqual(4));
 expect(releases).toHaveLength(4); expect(opened).toHaveBeenCalledTimes(4);
 releases.splice(0).forEach(r => r());
 await vi.waitFor(() => expect(releases.length).toBeGreaterThanOrEqual(4));
 expect(releases).toHaveLength(4); expect(opened).toHaveBeenCalledTimes(8);
 releases.splice(0).forEach(r => r());
 await vi.waitFor(() => expect(releases.length).toBeGreaterThanOrEqual(4)); releases.splice(0).forEach(r => r());
 expect(await Promise.all(requests)).toEqual(Array(12).fill('cache full'));
});

it('loops short reads and never parses zero padding after an early EOF', async () => {
 const raw = path.join(dir, 'short.RAF'); await fs.writeFile(raw, raf());
 const entry = { path: raw, name: 'short.RAF', size: 1024, mtimeMs: 1 }; const cache = createCache({ dir: path.join(dir, 'cache') });
 const original = fs.open.bind(fs); let calls = 0;
 vi.spyOn(fs, 'open').mockImplementation(async (...args) => { const handle = await original(...args); const read = handle.read.bind(handle);
   vi.spyOn(handle, 'read').mockImplementation((...args: any[]) => { calls++; return (read as any)(args[0], args[1], Math.min(args[2], 7), args[3]); }); return handle; });
 const result = await ensureHead(entry, cache); expect(result.metadata?.camera).toBe('Fujifilm X-T5'); expect(calls).toBeGreaterThan(2); expect(await fs.readFile(result.thumbnail!)).toEqual(Buffer.from([255,216,255,217]));
});
