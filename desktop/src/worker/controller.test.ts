import { EventEmitter } from 'node:events';
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { afterEach, expect, it } from 'vitest';
import { createWorkerController } from './controller';
import { createWorkerLibrary } from './library';
import { createCache } from './cache';
import { createListingAssembler, listingFrames } from '../protocol/listing';
import { isWorkerToMain, isListingFrame, isPortReply, isPortEvent } from '../protocol/rpc';
import { realisticFacts } from '../protocol/test-fixtures';
import type { LibraryPhoto } from '@/host';
class Port extends EventEmitter {
  peer?: Port; sent: any[] = []; closed = false;
  postMessage(...args: any[]) { expect(args).toHaveLength(1); this.sent.push(args[0]); this.peer?.emit('message', { data: structuredClone(args[0]) }); }
  start() {} close() { this.closed = true; }
}
function ports() { const a = new Port(), b = new Port(); a.peer = b; b.peer = a; return [a, b]; }
let dir: string | undefined;
afterEach(async () => { if (dir) await fs.rm(dir, { recursive: true, force: true }); dir = undefined; });
it('lists a real 2,000-photo library through main and window assemblers, with bounded frames on every hop', async () => {
  dir = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'xveon-protocol-')));
  const folder = path.join(dir, 'photos'), cacheDir = path.join(dir, 'cache'); await fs.mkdir(folder);
  const cache = createCache({ dir: cacheDir });
  const facts = realisticFacts(); Object.assign(facts.metadata!, { dateTime: '2026-09-30T18:30:00Z', iso: 800, exposureTime: 0.004 });
  for (let start = 0; start < 2000; start += 100) await Promise.all(Array.from({ length: 100 }, async (_, j) => {
    const name = `DSCF${start + j}.RAF`, raw = path.join(folder, name); await fs.writeFile(raw, 'RAW'); const stat = await fs.stat(raw);
    await cache.putFacts(cache.key({ path: raw, name, size: stat.size, mtimeMs: stat.mtimeMs }), facts);
  }));
  let windowPhotos: LibraryPhoto[] = []; const sizes: number[][] = [[], []]; const registry = new Map();
  const errors: string[] = []; const window = createListingAssembler((_f, photos) => { windowPhotos = photos; }, (_t, e) => errors.push(e));
  const [mainPort, windowPort] = ports(); windowPort.on('message', ({ data }) => { expect(isListingFrame(data)).toBe(true); expect(data.registry).toBeUndefined(); window.push(data); });
  const main = createListingAssembler((ref, photos, purpose) => {
    for (const frame of listingFrames({ token: 'window-list', activation: 'open', folder: ref, purpose }, photos)) { sizes[1].push(Buffer.byteLength(JSON.stringify(frame))); mainPort.postMessage(frame); }
  }, (_t, e) => errors.push(e));
  const [workerPort, parentPort] = ports(); parentPort.on('message', ({ data }) => {
    expect(isWorkerToMain(data)).toBe(true); sizes[0].push(Buffer.byteLength(JSON.stringify(data)));
    if (data.registry) for (const [id, raw] of data.registry) registry.set(id, raw);
    main.push(data);
  });
  const controller = createWorkerController({ postToMain: message => workerPort.postMessage(message), createLibrary: createWorkerLibrary });
  await controller.handleMain({ v: 1, kind: 'session', key: Buffer.from('key').toString('base64'), cacheDir });
  await controller.handleMain({ v: 1, kind: 'roots', realRoots: [folder] });
  await controller.handleMain({ v: 1, rid: 1, kind: 'list', path: folder, folderId: 'folder', token: 'open', activation: 'open', purpose: 'open' });
  expect(errors).toEqual([]); expect(windowPhotos).toHaveLength(2000); expect(registry.size).toBe(2000);
  expect(windowPhotos.map(p => p.originalName)).toEqual(Array.from({ length: 2000 }, (_, i) => `DSCF${i}.RAF`));
  expect(windowPhotos[1999].facts).toEqual(facts);
  expect(sizes.map(s => s.length)).toEqual([10, 10]); expect(sizes.flat().every(n => n <= 1_000_000)).toBe(true);
}, 30_000);
it('rejects malformed boundary data and invalid connect port counts', async () => {
  const sent: unknown[] = []; const c = createWorkerController({ postToMain: m => sent.push(m) }); const [a] = ports();
  await c.handleMain(null); await c.handleMain({ v: 9, kind: 'connect' }, [a]);
  await c.handleMain({ v: 1, kind: 'connect' }, []); await c.handleMain({ v: 1, kind: 'connect' }, [a, a]);
  expect(a.listenerCount('message')).toBe(0); expect(sent).toEqual([]);
});
it('cancels a listing while its library iterator is pending without emitting a partial result', async () => {
  let release!: () => void; const wait = new Promise<void>(r => { release = r; }); const messages: unknown[] = [];
  const c = createWorkerController({ postToMain: m => messages.push(m), createLibrary: () => ({ async *list() { await wait; yield { photos: [], registry: [] }; }, register() {}, setRoots() {}, saveEdit: async () => {}, saveFacts: async () => {}, thumbnail: async () => ({ path: null, facts: null }) }) });
  await c.handleMain({ v: 1, kind: 'session', key: 'a2V5', cacheDir: '/cache' });
  const pending = c.handleMain({ v: 1, rid: 1, kind: 'list', path: '/photos', folderId: 'f', token: 't', activation: 't', purpose: 'open' });
  await c.handleMain({ v: 1, kind: 'cancel-list', token: 't' }); release(); await pending;
  expect(messages).toEqual([]);
});
it('routes validated saves and rescans over the port while listings stay on parentPort', async () => {
  dir = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'xveon-port-'))); const folder = path.join(dir, 'photos'); await fs.mkdir(folder); await fs.writeFile(path.join(folder, 'a.RAF'), 'raw');
  const sent: any[] = []; const c = createWorkerController({ postToMain: m => sent.push(m) }); const [a, b] = ports();
  await c.handleMain({ v: 1, kind: 'session', key: 'a2V5', cacheDir: path.join(dir, 'cache') }); await c.handleMain({ v: 1, kind: 'roots', realRoots: [folder] });
  await c.handleMain({ v: 1, kind: 'connect' }, [a]); await c.handleMain({ v: 1, rid: 1, kind: 'list', path: folder, folderId: 'f', token: 't', activation: 't', purpose: 'open' });
  await c.handleMain({ v: 1, kind: 'watch', path: folder, folderId: 'f', activation: 't' });
  expect(sent.some(m => m.kind === 'listing-batch')).toBe(true);
  const photo = sent.find(m => m.kind === 'listing-batch').photos[0];
  const replies: any[] = []; b.on('message', ({ data }) => replies.push(data));
  b.postMessage(null); b.postMessage({ v: 2, rid: 2, op: 'rescan' });
  async function request(message: any) { const reply = new Promise<any>(resolve => { const listener = ({ data }: any) => { if (data.rid === message.rid) { b.off('message', listener); resolve(data); } }; b.on('message', listener); }); b.postMessage(message); return reply; }
  expect(await request({ v: 1, rid: 2, op: 'saveEdit', id: photo.id, edit: { ...photo.edit, preProcessOverrides: { exposure: 1 } } })).toEqual({ v: 1, rid: 2, ok: true });
  expect(await fs.readFile(path.join(folder, 'a.RAF.xmp'), 'utf8')).toContain('xveon:Exposure="1"');
  expect(await request({ v: 1, rid: 3, op: 'saveFacts', id: photo.id, facts: realisticFacts() })).toEqual({ v: 1, rid: 3, ok: true });
  expect(await request({ v: 1, rid: 4, op: 'rescan' })).toEqual({ v: 1, rid: 4, ok: true });
  expect(sent.filter(m => m.kind === 'listing-begin').at(-1)).toMatchObject({ purpose: 'replace', activation: 't' });
  await c.handleMain({ v: 1, rid: 5, kind: 'thumbnail', id: photo.id });
  expect(replies.some(isPortEvent)).toBe(true); expect(replies.every(m => isPortReply(m) || isPortEvent(m))).toBe(true);
  expect(await request({ v: 1, rid: 6, op: 'saveEdit', id: 'z'.repeat(22), edit: photo.edit })).toMatchObject({ ok: false, error: expect.stringMatching(/registered/) });
});
