import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { beforeEach, afterEach, it, expect, vi } from 'vitest';
import type { PhotoFacts } from '@/host';
import { createCache } from './cache';
import { listFolder } from './folder';
let dir: string;
const facts: PhotoFacts = { cfaType: null, metadata: null, resultMeta: null, resultMethod: null, lensProfile: null, status: 'queued', error: null };
beforeEach(async () => { dir = await fs.mkdtemp(path.join(os.tmpdir(), 'xveon-cache-')); });
afterEach(async () => { vi.restoreAllMocks(); await fs.rm(dir, { recursive: true, force: true }); });
it('invalidates facts when a RAW modification time changes', async () => { const raw = path.join(dir, 'a.RAF'); await fs.writeFile(raw, 'raw'); const cache = createCache({ dir: path.join(dir, 'cache') }); const key = cache.key((await listFolder(dir))[0]); await cache.putFacts(key, facts); expect(await cache.getFacts(key)).toEqual(facts); await fs.utimes(raw, new Date(0), new Date(1000)); expect(await cache.getFacts(cache.key((await listFolder(dir))[0]))).toBeNull(); });
it('evicts the least recently used whole entry below a 1 KB limit', async () => { const cache = createCache({ dir, limitBytes: 1024 }); const a = 'a'.repeat(64), b = 'b'.repeat(64); await cache.putThumb(a, new Uint8Array(600)); await new Promise(r => setTimeout(r, 10)); await cache.putThumb(b, new Uint8Array(600)); await new Promise(r => setTimeout(r, 10)); await cache.hasThumb(a); await cache.evict(); expect(await cache.hasThumb(a)).toBe(true); expect(await cache.hasThumb(b)).toBe(false); });
it('treats a deleted or corrupt cache as a miss and recreates it', async () => { const cache = createCache({ dir }); const key = 'a'.repeat(64); await cache.putFacts(key, facts); await fs.rm(dir, { recursive: true }); expect(await cache.getFacts(key)).toBeNull(); expect(await cache.hasThumb(key)).toBe(false); await cache.evict(); await cache.putFacts(key, facts); expect(await cache.getFacts(key)).toEqual(facts); await fs.writeFile(path.join(dir, key + '.json'), '{broken'); expect(await cache.getFacts(key)).toBeNull(); });
it('persists head completion separately from facts and forgets it when the cache disappears', async () => {
 const key = 'c'.repeat(64); const cache = createCache({ dir });
 expect(await cache.hasHead(key)).toBe(false); await cache.markHead(key);
 expect(await createCache({ dir }).hasHead(key)).toBe(true); expect(await cache.getFacts(key)).toBeNull();
 await fs.rm(dir, { recursive: true }); expect(await cache.hasHead(key)).toBe(false);
});

it.each(['head-first', 'facts-first'] as const)('preserves both owners during concurrent updates: %s', async (order) => {
  const key = 'd'.repeat(64);
  const cache = createCache({ dir });
  const other = createCache({ dir });
  const metadata = { camera: 'Fujifilm X-T5', lensModel: '', focalLength: 0, fNumber: 0 };
  const completed: PhotoFacts = { ...facts, status: 'done', cfaType: 'xtrans' };
  // Hold the first JSON rename so both requests are outstanding together.
  const rename = fs.rename.bind(fs);
  let entered!: () => void;
  const firstRename = new Promise<void>(resolve => { entered = resolve; });
  let release!: () => void;
  const gate = new Promise<void>(resolve => { release = resolve; });
  vi.spyOn(fs, 'rename').mockImplementationOnce(async (...args) => {
    entered(); await gate; await rename(...args);
  });
  const first = order === 'head-first' ? cache.putHeadMetadata(key, metadata) : cache.putFacts(key, completed);
  await firstRename;
  const second = order === 'head-first' ? other.putFacts(key, completed) : other.putHeadMetadata(key, metadata);
  release();
  await Promise.all([first, second]);
  expect(await createCache({ dir }).getFacts(key)).toEqual({ ...completed, metadata });
});

it('coalesces eviction after writes and never schedules scans for cache hits', async () => {
 vi.useFakeTimers(); const cache = createCache({ dir }); const evict = vi.spyOn(cache, 'evict').mockResolvedValue(); const key = 'e'.repeat(64);
 try { await Promise.all([cache.putFacts(key, facts), cache.putThumb(key, new Uint8Array(3)), cache.markHead(key)]);
 expect(evict).not.toHaveBeenCalled(); await vi.advanceTimersByTimeAsync(1000); expect(evict).toHaveBeenCalledOnce();
 await Promise.all(Array.from({ length: 150 }, () => cache.hasHead(key))); await vi.advanceTimersByTimeAsync(60_000); expect(evict).toHaveBeenCalledOnce(); }
 finally { vi.useRealTimers(); }
});
