import { EventEmitter } from 'node:events';
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { afterEach, expect, it, vi } from 'vitest';
import { createFolderStore } from './folders';
import { createWorkerSupervisor } from './worker';
import { createFolderRequests } from './folder-requests';
import { listingFrames } from '../protocol/listing';
import { photo } from '../protocol/test-fixtures';
class Child extends EventEmitter {
  pid = 1; sent: any[] = [];
  postMessage(message: unknown) { this.sent.push(message); }
  kill() { return true; }
}
const cleanups: (() => Promise<void>)[] = [];
afterEach(async () => { vi.restoreAllMocks(); for (const cleanup of cleanups.splice(0)) await cleanup(); });
function held<T>() { let resolve!: (value: T) => void; const promise = new Promise<T>(r => { resolve = r; }); return { promise, resolve }; }
async function harness() {
  const dir = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'requests-'))); const A = path.join(dir, 'A'), B = path.join(dir, 'B'); await fs.mkdir(A); await fs.mkdir(B);
  const store = createFolderStore(path.join(dir, 'folders.json')); const child = new Child();
  const worker = createWorkerSupervisor({ fork: () => child, sessionKey: Buffer.alloc(32), cacheDir: dir, onEvent() {} }); await worker.ready();
  cleanups.push(async () => { worker.stop(); await store.flush(); await fs.rm(dir, { recursive: true, force: true }); });
  let chosen = A; const frames: any[] = [], error = vi.fn(); const requests = createFolderRequests({ store, worker, chooseFolder: async () => chosen, send: f => frames.push(f), error, accepted() {} });
  const choose = (value: string) => { chosen = value; return requests.openFolder(); };
  async function listed(index: number) { await vi.waitFor(() => expect(child.sent.filter(m => m.kind === 'list').length).toBeGreaterThan(index)); return child.sent.filter(m => m.kind === 'list')[index]; }
  function complete(request: any, name: string) {
    const p = { ...photo(), id: name.repeat(22) }; const folder = { id: request.folderId, name: path.basename(request.path) };
    for (const frame of listingFrames({ token: request.token, activation: request.activation, purpose: request.purpose, folder, stamp: request.stamp }, [p], [[p.id, path.join(request.path, name + '.RAF')]])) child.emit('message', frame);
  }
  return { A, B, dir, store, worker, child, requests, choose, frames, error, listed, complete };
}
it('latest listing wins; A resolves null and late A cannot change any committed state', async () => {
  const h = await harness(); const a = h.choose(h.A); const first = await h.listed(0); const b = h.choose(h.B); const second = await h.listed(1);
  await expect(a).resolves.toBeNull(); h.complete(second, 'b'); expect(await b).toEqual({ token: second.token }); h.complete(first, 'a');
  expect(h.worker.current).toEqual({ path: h.B, folderId: second.folderId, activation: second.activation }); expect(h.store.last()?.name).toBe('B'); expect(JSON.parse(await fs.readFile(path.join(h.dir, 'folders.json'), 'utf8')).last).toBe(second.folderId);
  expect(h.worker.roots).toEqual([h.B]); expect(h.child.sent.filter(m => m.kind === 'watch').map(m => m.path)).toEqual([h.B]); expect(h.worker.registry.has('a'.repeat(22))).toBe(false); expect(h.frames.every(f => f.token === second.token)).toBe(true); expect(h.child.sent).toContainEqual({ v: 1, kind: 'cancel-list', token: first.token });
});
it('A → B → A rejects first-visit replacements and preserves the final watcher', async () => {
  const h = await harness(); const a1 = h.choose(h.A); const first = await h.listed(0); h.complete(first, 'a'); await a1;
  const b = h.choose(h.B); const second = await h.listed(1); h.complete(second, 'b'); await b;
  const a2 = h.choose(h.A); const third = await h.listed(2); h.complete(third, 'c'); await a2; h.frames.length = 0;
  h.complete({ ...first, purpose: 'replace', token: 'late-first' }, 'd');
  expect(h.frames).toEqual([]); expect(h.worker.registry.has('d'.repeat(22))).toBe(false); expect(h.worker.current?.activation).toBe(third.activation); expect(h.child.sent.filter(m => m.kind === 'watch').at(-1).activation).toBe(third.activation);
  h.complete({ ...third, purpose: 'replace', token: 'valid-replace' }, 'e'); expect(h.frames.at(-1)?.token).toBe('valid-replace'); expect(h.frames.some(f => 'registry' in f)).toBe(false);
});
it('cancels during prepared-worker readiness without committing the older listing', async () => {
  const h = await harness(); const gate = held<void>(); const prepare = h.worker.prepareCommit.bind(h.worker);
  vi.spyOn(h.worker, 'prepareCommit').mockImplementationOnce(async () => { await gate.promise; return prepare(); });
  const a = h.choose(h.A); const first = await h.listed(0); h.complete(first, 'a'); await vi.waitFor(() => expect(h.worker.prepareCommit).toHaveBeenCalled());
  const b = h.choose(h.B); const second = await h.listed(1); h.complete(second, 'b'); await b; gate.resolve(); await expect(a).resolves.toBeNull(); await Promise.resolve();
  expect(h.store.recent().map(f => f.name)).toEqual(['B']); expect(h.worker.roots).toEqual([h.B]);
});
it('superseding a request awaiting persistence resolves it null and publishes only newest frames', async () => {
  const h = await harness(); const gate = held<void>(); const flush = h.store.flush.bind(h.store);
  vi.spyOn(h.store, 'flush').mockImplementationOnce(async () => { await gate.promise; await flush(); });
  const a = h.choose(h.A); const first = await h.listed(0); h.complete(first, 'a'); await vi.waitFor(() => expect(h.store.flush).toHaveBeenCalled());
  const b = h.choose(h.B); const second = await h.listed(1); h.complete(second, 'b'); await b; await expect(a).resolves.toBeNull(); gate.resolve(); await Promise.resolve();
  expect(h.store.last()?.name).toBe('B'); expect(h.frames.every(f => f.token === second.token)).toBe(true);
});
it('drops open the first RAW containing folder and select only dropped files in it', async () => {
  const h = await harness(); await fs.writeFile(path.join(h.A, 'a.RAF'), 'raw'); await fs.writeFile(path.join(h.B, 'b.RAF'), 'raw');
  const opening = h.requests.openDropped([path.join(h.dir, 'missing'), path.join(h.A, 'a.RAF'), path.join(h.B, 'b.RAF')]); const request = await h.listed(0); h.complete(request, 'a');
  expect(await opening).toEqual({ token: request.token, selected: ['a'.repeat(22)] }); expect(h.worker.current?.path).toBe(h.A);
});
it('reports failed recent-folder persistence but returns the accepted folder consistently', async () => {
  const h = await harness(); vi.spyOn(h.store, 'flush').mockRejectedValueOnce(new Error('ENOSPC: disk full'));
  const opening = h.choose(h.B); const request = await h.listed(0); h.complete(request, 'b');
  expect(await opening).toEqual({ token: request.token }); expect(h.frames.at(-1).token).toBe(request.token); expect(h.worker.current?.path).toBe(h.B);
  expect(h.error).toHaveBeenCalledWith('B', 'ENOSPC: disk full');
});
it('publishes open before the newest replacement completed while folder persistence is held', async () => {
  const h = await harness(); const gate = held<void>(); const flush = h.store.flush.bind(h.store);
  vi.spyOn(h.store, 'flush').mockImplementationOnce(async () => { await gate.promise; await flush(); });
  const opening = h.choose(h.A); const request = await h.listed(0); h.complete(request, 'a'); await vi.waitFor(() => expect(h.store.flush).toHaveBeenCalled());
  h.complete({ ...request, purpose: 'replace', token: 'replacement-1' }, 'b');
  h.complete({ ...request, purpose: 'replace', token: 'replacement-2' }, 'c');
  expect(h.frames).toEqual([]);
  gate.resolve(); expect(await opening).toEqual({ token: request.token });
  expect(h.frames.filter(f => f.kind === 'listing-begin').map(f => [f.purpose, f.token])).toEqual([['open', request.token], ['replace', 'replacement-2']]);
  expect(h.frames.filter(f => f.kind === 'listing-batch').at(-1).photos[0].id).toBe('c'.repeat(22));
});
it('reports an unpublishable replacement instead of throwing from the worker message callback', async () => {
  const h = await harness(); const opening = h.choose(h.A); const request = await h.listed(0); h.complete(request, 'a'); await opening; h.frames.length = 0;
  const huge = { ...photo(), editingNote: '' };
  const batch = { v: 1 as const, kind: 'listing-batch' as const, token: 'huge', seq: 0, photos: [huge] };
  huge.editingNote = 'x'.repeat(999_990 - Buffer.byteLength(JSON.stringify(batch)));
  const frames = [{ v: 1, kind: 'listing-begin', token: 'huge', activation: request.activation, folder: { id: request.folderId, name: 'A' }, purpose: 'replace', total: 1 }, batch, { v: 1, kind: 'listing-end', token: 'huge', total: 1 }];
  expect(() => { for (const frame of frames) h.child.emit('message', frame); }).not.toThrow();
  expect(h.frames).toEqual([]); expect(h.error).toHaveBeenCalledWith('A', expect.stringContaining('listing message limit'));
});

it('preserves worker revision and scan stamps when main republishes open and replacement listings', async () => {
  const h = await harness(); const opening = h.choose(h.A); const request = await h.listed(0);
  const stamp = { worker: h.worker.instance, revision: 4, scan: 1 };
  h.complete({ ...request, stamp }, 'a'); await opening;
  expect(h.frames[0].stamp).toEqual(stamp); h.frames.length = 0;
  const replacementStamp = { ...stamp, revision: 6, scan: 2 };
  h.complete({ ...request, token: 'replacement', purpose: 'replace', stamp: replacementStamp }, 'b');
  expect(h.frames[0].stamp).toEqual(replacementStamp);
});
