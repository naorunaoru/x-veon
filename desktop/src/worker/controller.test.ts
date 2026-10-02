import { EventEmitter } from 'node:events';
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { afterEach, expect, it, vi } from 'vitest';
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
  expect(await request({ v: 1, rid: 2, op: 'saveEdit', id: photo.id, edit: { ...photo.edit, preProcessOverrides: { exposure: 1 } } })).toEqual({ v: 1, rid: 2, ok: true, stamp: { worker: expect.any(String), revision: 2 } });
  expect(await fs.readFile(path.join(folder, 'a.RAF.xmp'), 'utf8')).toContain('xveon:Exposure="1"');
  expect(await request({ v: 1, rid: 3, op: 'saveFacts', id: photo.id, facts: realisticFacts() })).toEqual({ v: 1, rid: 3, ok: true, stamp: { worker: expect.any(String), revision: 4 } });
  expect(await request({ v: 1, rid: 4, op: 'rescan' })).toEqual({ v: 1, rid: 4, ok: true });
  expect(sent.filter(m => m.kind === 'listing-begin').at(-1)).toMatchObject({ purpose: 'replace', activation: 't' });
  await c.handleMain({ v: 1, rid: 5, kind: 'thumbnail', id: photo.id });
  expect(replies.some(isPortEvent)).toBe(true); expect(replies.every(m => isPortReply(m) || isPortEvent(m))).toBe(true);
  expect(await request({ v: 1, rid: 6, op: 'saveEdit', id: 'z'.repeat(22), edit: photo.edit })).toMatchObject({ ok: false, error: expect.stringMatching(/registered/) });
});

function heldListings() {
  const scans: { release(): void }[] = [];
  let edit = defaultPhotoEdit();
  let facts = fakePhoto().facts;
  const published: LibraryPhoto[][] = [];
  const assembler = createListingAssembler((_folder, photos) => published.push(photos), (_token, error) => { throw new Error(error); });
  const controller = createWorkerController({ postToMain: frame => { if (isListingFrame(frame)) assembler.push(frame); }, createLibrary: () => ({
    async *list() {
      const photo = { ...fakePhoto('a'.repeat(22)), edit: structuredClone(edit), facts: structuredClone(facts) };
      await new Promise<void>(release => scans.push({ release }));
      yield { photos: [photo], registry: [[photo.id, '/photos/a.RAF']] as [string, string][] };
    }, register() {}, setRoots() {}, async saveEdit(_id, next) { edit = next; },
    async saveFacts(_id, next) { facts = next; }, async thumbnail() { facts = { ...facts, metadata: { camera: 'extracted', lensModel: '', focalLength: 0, fNumber: 0 } }; return { path: null, facts }; },
  }) });
  return { controller, scans, published, setEdit(next: typeof edit) { edit = next; } };
}
import { defaultPhotoEdit } from '@/app/photo-edit';
import { fakePhoto, fakeHost } from '@/test/fake-host';
import { useAppStore } from '@/app/store';
import { fromLibraryPhoto } from '@/app/store/photo';
import { setHost } from '@/app/services/host';
import { startPersistence, flushPersistence, unsavedEdits, cancelPhotoSave } from '@/app/services/persistence';
vi.mock('@/app/storage/settings-storage', () => ({ putSetting: vi.fn(), pauseSettings: vi.fn(), resumeSettings: vi.fn() }));
const tick = async () => { for (let i = 0; i < 20; i++) await Promise.resolve(); };
async function initHeld(h: ReturnType<typeof heldListings>) {
  await h.controller.handleMain({ v: 1, kind: 'session', key: 'a2V5', cacheDir: '/cache' });
  await h.controller.handleMain({ v: 1, kind: 'watch', path: '/photos', folderId: 'f', activation: 'a' });
}
it.each(['watch', 'restore'] as const)('supersedes an older same-activation %s scan that finishes after the newer scan', async origin => {
  const h = heldListings(); await initHeld(h);
  const first = origin === 'watch' ? h.controller.replace() : h.controller.handleMain({ v: 1, rid: 9, kind: 'list', path: '/photos', folderId: 'f', token: 'restored', activation: 'a', purpose: 'replace' }); await tick();
  h.setEdit({ ...defaultPhotoEdit(), preProcessOverrides: { exposure: 2 } });
  const second = h.controller.replace(); await tick();
  h.scans[1].release(); await second; h.scans[0].release(); await first;
  expect(h.published.map(p => p[0].edit.preProcessOverrides.exposure)).toEqual([2]);
});
it('rereads a held pre-edit snapshot after the real ledger has acknowledged the save', async () => {
  const h = heldListings(); await initHeld(h);
  const [a, b] = ports(); await h.controller.handleMain({ v: 1, kind: 'connect' }, [a]);
  const host = fakeHost(); let rid = 0;
  host.library.save = async (id, edit) => {
    const requestId = ++rid;
    await new Promise<void>((resolve, reject) => {
      const listener = ({ data }: any) => { if (data.rid !== requestId) return; b.off('message', listener); if (data.ok) resolve(); else reject(new Error(data.error)); };
      b.on('message', listener); b.postMessage({ v: 1, rid: requestId, op: 'saveEdit', id, edit });
    });
  };
  setHost(host);
  const id = 'a'.repeat(22);
  useAppStore.setState({ files: [fromLibraryPhoto(fakePhoto(id))], demosaicMethod: 'dht', folder: { id: 'f', name: 'f' } });
  vi.stubGlobal('window', new EventTarget());
  const stop = startPersistence();
  try {
    const scan = h.controller.replace(); await tick();
    useAppStore.getState().setFileLookPreset(id, 'umbra');
    await flushPersistence(); expect(unsavedEdits()).toEqual([]);
    h.scans[0].release(); await tick();
    // An invalidated scan must reread, never publish its old sidecar after acknowledgment.
    expect(h.published).toEqual([]);
    expect(h.scans).toHaveLength(2);
    h.scans[1].release(); await scan;
    expect(h.published[0][0].edit.lookPreset).toBe('umbra');
  } finally { stop(); await cancelPhotoSave(id); vi.unstubAllGlobals(); }
});

it.each(['saveFacts'] as const)('rereads cached facts when %s completes during a held listing', async operation => {
  const h = heldListings(); await initHeld(h);
  const [a, b] = ports(); await h.controller.handleMain({ v: 1, kind: 'connect' }, [a]);
  const scan = h.controller.replace(); await tick();
  {
    b.postMessage({ v: 1, rid: 9, op: 'saveFacts', id: 'a'.repeat(22), facts: { ...fakePhoto().facts, metadata: { camera: 'saved', lensModel: '', focalLength: 0, fNumber: 0 } } });
    await tick();
  }
  h.scans[0].release(); await tick();
  expect(h.published).toEqual([]); expect(h.scans).toHaveLength(2);
  h.scans[1].release(); await scan;
  expect(h.published[0][0].facts.metadata?.camera).toBe('saved');
});

it.each(['thumbnail', 'saveEdit'] as const)('lists B promptly while unrelated A %s is held', async operation => {
 let release!: () => void; const gate = new Promise<void>(r => { release = r; }); const started = vi.fn();
 const messages: any[] = [];
 const c = createWorkerController({ postToMain: m => messages.push(m), createLibrary: () => ({
   async *list() { yield { photos: [], registry: [] }; }, register() {}, setRoots() {},
   async saveEdit() { started(); await gate; }, async saveFacts() {}, async thumbnail() { started(); await gate; return { path: null, facts: null }; },
 }) });
 await c.handleMain({ v: 1, kind: 'session', key: 'a2V5', cacheDir: '/cache' });
 await c.handleMain({ v: 1, kind: 'register', entries: [['a'.repeat(22), '/A/a.RAF']] });
 const [a, b] = ports(); await c.handleMain({ v: 1, kind: 'connect' }, [a]);
 const pending = operation === 'thumbnail' ? c.handleMain({ v: 1, rid: 1, kind: 'thumbnail', id: 'a'.repeat(22) }) : Promise.resolve(b.postMessage({ v: 1, rid: 1, op: 'saveEdit', id: 'a'.repeat(22), edit: defaultPhotoEdit() }));
 await tick(); expect(started).toHaveBeenCalledOnce();
 const listing = c.handleMain({ v: 1, rid: 2, kind: 'list', path: '/B', folderId: 'B', token: 'B', activation: 'B', purpose: 'open' });
 await tick();
 try { expect(messages.some(m => m.kind === 'listing-end' && m.token === 'B')).toBe(true); }
 finally { release(); await pending; await listing; }
});

it('folds completed thumbnail metadata into a held listing without restarting its scan', async () => {
 const h = heldListings(); await initHeld(h);
 const scan = h.controller.replace(); await tick();
 await h.controller.handleMain({ v: 1, rid: 9, kind: 'thumbnail', id: 'a'.repeat(22) });
 h.scans[0].release(); await scan;
 expect(h.scans).toHaveLength(1); expect(h.published[0][0].facts.metadata?.camera).toBe('extracted');
});

it('does not carry thumbnail metadata across a RAW source revision', async () => {
 const messages: any[] = []; const item = { ...fakePhoto('a'.repeat(22)), sourceVersion: 'b'.repeat(64) };
 const c = createWorkerController({ postToMain: m => messages.push(m), createLibrary: () => ({
   async *list() { yield { photos: [item], registry: [[item.id, '/photos/a.RAF']] as [string, string][] }; }, register() {}, setRoots() {}, async saveEdit() {}, async saveFacts() {},
   async thumbnail() { return { path: null, sourceVersion: 'a'.repeat(64), facts: { ...item.facts, metadata: { camera: 'old RAW', lensModel: '', focalLength: 0, fNumber: 0 } } }; },
 }) });
 await c.handleMain({ v: 1, kind: 'session', key: 'a2V5', cacheDir: '/cache' });
 await c.handleMain({ v: 1, rid: 1, kind: 'thumbnail', id: item.id });
 await c.handleMain({ v: 1, rid: 2, kind: 'list', path: '/photos', folderId: 'f', token: 't', activation: 't', purpose: 'open' });
 expect(messages.find(m => m.kind === 'listing-batch').photos[0].facts.metadata).toBeNull();
});
