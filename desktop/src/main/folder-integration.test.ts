import { EventEmitter } from 'node:events';
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { afterEach, expect, it, vi } from 'vitest';
import type { DesktopBridge, BridgeEvent } from '../protocol/bridge';
import type { PortRequest } from '../protocol/rpc';
import { createDesktopHost } from '../host';
import { createFolderStore } from './folders';
import { createFolderRequests } from './folder-requests';
import { createWorkerSupervisor } from './worker';
import { buildMenuTemplate } from './menu';
import { listingFrames } from '../protocol/listing';
import { fakePhoto } from '@/test/fake-host';
import { setHost } from '@/app/services/host';
import { useAppStore } from '@/app/store';
import { fromLibraryPhoto } from '@/app/store/photo';
import { openFolder, startLibraryWatching } from '@/app/services/library';
import { initApp, startHostCoordination } from '@/app/services/bootstrap';
import { startPersistence, unsavedEdits, cancelPhotoSave } from '@/app/services/persistence';
import { shouldAutoProcess } from '@/app/hooks/useAutoProcess';
vi.mock('@/app/services/processing', () => ({ discardResult: vi.fn(), setPipeline: vi.fn() }));
vi.mock('node:fs/promises', async importOriginal => { const original = await importOriginal<typeof import('node:fs/promises') & { default: typeof fs }>(); return { ...original, realpath: (...args: Parameters<typeof fs.realpath>) => original.default.realpath(...args) }; });
vi.mock('@/pipeline', () => ({ initPipeline: vi.fn(async () => ({ models: { availableSizes: () => new Set(['S']), backend: 'test' } })) }));
vi.mock('@/app/lens/lensfun', () => ({ matchLens: vi.fn(async () => null) }));
vi.mock('@/app/storage/settings-storage', () => ({ getSetting: vi.fn(async () => undefined), putSetting: vi.fn(async () => {}), pauseSettings: vi.fn(), resumeSettings: vi.fn() }));
const tick = async () => { for (let i = 0; i < 80; i++) await Promise.resolve(); };
function held<T>() { let resolve!: (value: T) => void; const promise = new Promise<T>(r => { resolve = r; }); return { promise, resolve }; }
class Child extends EventEmitter {
  pid = 1; sent: any[] = [];
  postMessage(message: unknown) { this.sent.push(message); }
  kill() { return true; }
}
class Port extends EventTarget {
  sent: PortRequest[] = []; hold = true;
  start() {} close() {}
  postMessage(message: PortRequest) { this.sent.push(message); if (!this.hold) queueMicrotask(() => this.confirm(message)); }
  confirm(request: PortRequest) { this.dispatchEvent(new MessageEvent('message', { data: { v: 1, rid: request.rid, ok: true } })); }
}
const cleanups: (() => Promise<void> | void)[] = [];
afterEach(async () => { for (const cleanup of cleanups.splice(0).reverse()) await cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); setHost(null); });
async function harness() {
  const dir = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'folder-integration-')));
  const A = path.join(dir, 'A'), B = path.join(dir, 'B'); await fs.mkdir(A); await fs.mkdir(B);
  const store = createFolderStore(path.join(dir, 'folders.json')); const recent = store.remember(B); await store.flush();
  const child = new Child(), worker = createWorkerSupervisor({ fork: () => child, sessionKey: Buffer.alloc(32), cacheDir: dir, onEvent() {} }); await worker.ready();
  const win = new EventTarget(), port = new Port(), events = new Set<(event: BridgeEvent) => void>();
  vi.stubGlobal('window', win); vi.stubGlobal('location', { origin: 'app://bundle' }); vi.stubGlobal('matchMedia', () => ({ matches: false }));
  const emit = (event: BridgeEvent) => { for (const receive of events) receive(event); };
  const picker = vi.fn(async () => A as string | null);
  const requests = createFolderRequests({ store, worker, chooseFolder: picker, send: frame => emit({ kind: 'listing', frame }), accepted() {}, error: vi.fn() });
  const bridge: DesktopBridge = { version: 2, chooseExportDestination: async () => null, revealExport: async () => {}, loadLast: requests.loadLast, openFolder: vi.fn(requests.openFolder), openDropped: async () => null, recentFolders: async () => store.recent(), displayReadings: async () => null, checkForUpdate: async () => null,
    requestWorkerPort: async requestId => { const event = new MessageEvent('message', { data: { type: 'xveon-port', version: 2, requestId }, origin: 'app://bundle' }); Object.defineProperties(event, { source: { value: win }, ports: { value: [port] } }); win.dispatchEvent(event); },
    updateUnsaved() {}, respondFlush() {}, onEvent: receive => { events.add(receive); return () => events.delete(receive); } };
  const host = createDesktopHost(bridge); setHost(host);
  const initial = fakePhoto('c'.repeat(22)); initial.edit.demosaicMethod = 'dht';
  useAppStore.setState({ files: [fromLibraryPhoto(initial)], folder: null, selectedFileId: initial.id, initialized: false, initError: null, processingFileId: null });
  const stopCoordination = startHostCoordination(), stopWatching = startLibraryWatching(), stopPersistence = startPersistence();
  cleanups.push(async () => { port.hold = false; for (const message of port.sent) port.confirm(message); await Promise.all(unsavedEdits().map(e => cancelPhotoSave(e.id))); stopPersistence(); stopWatching(); stopCoordination(); worker.stop(); await store.flush(); await fs.rm(dir, { recursive: true, force: true }); });
  const fileMenu = buildMenuTemplate(store.recent(), emit, 'darwin').find(item => item.label === 'File')!.submenu as any[];
  async function listed(index: number) { await vi.waitFor(() => expect(child.sent.filter(m => m.kind === 'list').length).toBeGreaterThan(index)); return child.sent.filter(m => m.kind === 'list')[index]; }
  function complete(request: any) {
    const p = fakePhoto(path.basename(request.path).toLowerCase().repeat(22));
    for (const frame of listingFrames({ token: request.token, activation: request.activation, purpose: request.purpose, folder: { id: request.folderId, name: path.basename(request.path) } }, [p], [[p.id, path.join(request.path, p.originalName)]])) child.emit('message', frame);
  }
  return { A, B, dir, store, worker, child, host, port, picker, requests, recent, listed, complete, bridge, nativeOpen: () => fileMenu[0].click() };
}
it('keeps shared renderer, main, last folder and watcher on recent B after a native open overlaps a real save flush', async () => {
  const h = await harness(); const selection = held<string | null>(); h.picker.mockImplementationOnce(() => selection.promise);
  useAppStore.getState().setFilePreProcessOverride('c'.repeat(22), 'exposure', 1);
  h.nativeOpen(); await tick(); expect(h.port.sent.map(m => m.op)).toEqual(['saveEdit']);
  useAppStore.getState().setFilePreProcessOverride('c'.repeat(22), 'exposure', 2);
  const recent = openFolder(h.recent); await tick(); expect(h.child.sent.some(m => m.kind === 'list')).toBe(false);
  h.port.confirm(h.port.sent[0]); await tick();
  const edits = h.port.sent.filter(m => m.op === 'saveEdit'); expect(edits).toHaveLength(2);
  h.port.hold = false; for (const request of h.port.sent) h.port.confirm(request);
  const request = await h.listed(0); h.complete(request); await recent;
  selection.resolve(h.A); await tick();
  // The stale native picker must not start another list even if it resolves late.
  await fs.stat(h.B); await tick();
  for (const late of h.child.sent.filter(m => m.kind === 'list' && m.path === h.A)) h.complete(late);
  await tick(); await h.store.flush();
  expect(vi.mocked(h.bridge.openFolder).mock.calls.map(call => call[0])).toEqual([undefined, h.recent.id]);
  expect(useAppStore.getState().folder?.name).toBe('B');
  expect(h.worker.current?.path).toBe(h.B); expect(h.store.last()?.name).toBe('B');
  expect(JSON.parse(await fs.readFile(path.join(h.dir, 'folders.json'), 'utf8')).last).toBe(h.recent.id);
  expect(h.child.sent.filter(m => m.kind === 'watch').at(-1).path).toBe(h.B);
});
it('finishes initialization and permits processing when accepted B supersedes unresolved startup realpath', async () => {
  const h = await harness(); h.store.remember(h.A); await h.store.flush();
  const gate = held<string>(), entered = held<void>(), original = fs.realpath.bind(fs);
  vi.spyOn(fs, 'realpath').mockImplementation(async target => { if (target === h.A) { entered.resolve(); return gate.promise; } return original(target); });
  let initialized = false; const init = initApp({ cancelled: false }).then(() => { initialized = true; }); await entered.promise;
  const alternate = openFolder(h.recent); const request = await h.listed(0); h.complete(request); await alternate; await tick();
  try {
    expect(useAppStore.getState().initError).toBeNull(); expect(initialized).toBe(true); expect(useAppStore.getState().folder?.name).toBe('B');
    const state = useAppStore.getState(); expect(shouldAutoProcess(state.files[0], state.initialized, state.processingFileId !== null)).toBe(true);
  } finally { gate.resolve(h.A); await tick(); for (const late of h.child.sent.filter(m => m.kind === 'list' && m.path === h.A)) h.complete(late); await init; await tick(); }
  expect(h.child.sent.filter(m => m.kind === 'list')).toHaveLength(1); expect(h.worker.current?.path).toBe(h.B); expect(h.store.last()?.name).toBe('B');
});

it.each(['cancelled-picker', 'invalid-recent'])('keeps the earlier valid native choice after a real save flush and a newer %s', async kind => {
  const h = await harness(); const selection = held<string | null>();
  h.picker.mockImplementationOnce(() => selection.promise).mockResolvedValueOnce(null);
  useAppStore.getState().setFilePreProcessOverride('c'.repeat(22), 'exposure', 1);
  h.nativeOpen(); await tick();
  useAppStore.getState().setFilePreProcessOverride('c'.repeat(22), 'exposure', 2);
  const later = kind === 'cancelled-picker' ? openFolder() : openFolder({ id: 'missing', name: 'Missing' }); await tick();
  h.port.confirm(h.port.sent[0]); await tick();
  h.port.hold = false; for (const request of h.port.sent) h.port.confirm(request);
  await later; selection.resolve(h.A);
  const request = await h.listed(0); h.complete(request); await tick(); await h.store.flush();
  expect(useAppStore.getState().folder?.name).toBe('A'); expect(h.worker.current?.path).toBe(h.A);
  expect(h.store.last()?.name).toBe('A'); expect(h.child.sent.filter(m => m.kind === 'watch').at(-1).path).toBe(h.A);
  expect(unsavedEdits()).toEqual([]);
});
