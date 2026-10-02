import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { createDesktopHost } from './index';
import type { DesktopBridge, BridgeEvent } from '../protocol/bridge';
import type { LibraryPhoto, Host, UnsavedSummary } from '@/host';
import type { PortRequest } from '../protocol/rpc';
import { listingFrames } from '../protocol/listing';
import { fakePhoto } from '@/test/fake-host';
vi.mock('@/app/services/processing', () => ({ discardResult: vi.fn() }));
vi.mock('@/app/lens/lensfun', () => ({ matchLens: vi.fn(async () => null) }));
vi.mock('@/app/storage/settings-storage', () => ({ getSetting: vi.fn(), putSetting: vi.fn(async () => {}), pauseSettings: vi.fn(), resumeSettings: vi.fn() }));
import { useAppStore } from '@/app/store';
import { fromLibraryPhoto } from '@/app/store/photo';
import { setHost } from '@/app/services/host';
import { switchFolder, openFolder, importFiles } from '@/app/services/library';
import { startPersistence, flushPersistence, unsavedEdits, onUnsavedChange, cancelPhotoSave } from '@/app/services/persistence';
const id = 'a'.repeat(22), folder = { id: 'A', name: 'A' };
const photo = (): LibraryPhoto => ({ ...fakePhoto(id), name: 'photo.RAF', edit: { ...fakePhoto(id).edit, demosaicMethod: 'dht' } });
class FakePort extends EventTarget {
  sent: PortRequest[] = [];
  closed = false;
  sidecar: unknown;
  hold = false;
  error: string | null = null;
  start() {}
  close() { this.closed = true; }
  receive(data: unknown) { this.dispatchEvent(new MessageEvent('message', { data })); }
  postMessage(request: PortRequest, transfer?: unknown) {
    if (this.closed) throw new Error('Port is closed');
    expect(transfer).toBeUndefined();
    this.sent.push(request);
    if (!this.hold) queueMicrotask(() => this.confirm(request));
  }
  confirm(request: PortRequest) {
    if (!this.error && request.op === 'saveEdit') this.sidecar = request.edit;
    this.receive(this.error ? { v: 1, rid: request.rid, ok: false, error: this.error } : { v: 1, rid: request.rid, ok: true });
  }
}
let bridge: DesktopBridge, host: Host, win: EventTarget, ports: FakePort[], listeners: Set<(e: BridgeEvent) => void>, stop = () => {};
const emit = (event: BridgeEvent) => { for (const listener of listeners) listener(event); };
const tick = async () => { for (let n = 0; n < 20; n++) await Promise.resolve(); };
function frames(token: string, activation: string, purpose: 'open' | 'replace' = 'open', photos = [photo()], f = folder) {
  for (const frame of listingFrames({ token, activation, purpose, folder: f }, photos)) emit({ kind: 'listing', frame });
}
async function accept(token: string, activation: string, f = folder) {
  vi.mocked(bridge.openFolder).mockImplementationOnce(async () => { frames(token, activation, 'open', [photo()], f); return { token }; });
  return host.library.openFolder!(f);
}
beforeEach(() => {
  win = new EventTarget(); ports = []; listeners = new Set(); stop = () => {};
  vi.stubGlobal('window', win); vi.stubGlobal('location', { origin: 'app://bundle' }); vi.stubGlobal('matchMedia', (query: string) => ({ matches: query === '(dynamic-range: high)' }));
  bridge = { version: 2, loadLast: vi.fn(async () => null), openFolder: vi.fn(async () => null), openDropped: vi.fn(async () => null), recentFolders: vi.fn(async () => [folder]),
    requestWorkerPort: vi.fn(async (requestId: string) => { const port = new FakePort(); ports.push(port); const e = new MessageEvent('message', { data: { type: 'xveon-port', version: 2, requestId }, origin: 'app://bundle' }); Object.defineProperty(e, 'source', { value: win }); Object.defineProperty(e, 'ports', { value: [port] }); win.dispatchEvent(e); }),
    updateUnsaved: vi.fn(), respondFlush: vi.fn(), onEvent: listener => { listeners.add(listener); return () => listeners.delete(listener); } };
  host = createDesktopHost(bridge);
});
afterEach(async () => { if (host.library) await Promise.all(unsavedEdits().map(e => cancelPhotoSave(e.id))); stop(); vi.useRealTimers(); vi.unstubAllGlobals(); });
it('sends facts without an edit and starts best-effort facts caching after an edit acknowledgement', async () => {
  expect(host.library).toBeDefined(); const p = photo();
  await host.library.saveFacts(id, p.facts);
  expect(ports[0].sent).toEqual([{ v: 1, rid: expect.any(Number), op: 'saveFacts', id, facts: p.facts }]);
  await host.library.save(id, p.edit, p.facts);
  expect(ports[0].sent.map(r => r.op)).toEqual(['saveFacts', 'saveEdit', 'saveFacts']);
  ports[0].error = 'ENOSPC: no space left on device';
  await expect(host.library.save(id, p.edit, p.facts)).rejects.toThrow('ENOSPC: no space left on device');
});
it('cannot overwrite a held newer edit with concurrently saved facts', async () => {
  expect(host.library).toBeDefined(); await host.library.saveFacts(id, photo().facts); ports[0].sent = []; ports[0].hold = true;
  const edit = { ...photo().edit, preProcessOverrides: { exposure: 2 } };
  const saving = host.library.save(id, edit, photo().facts); await tick();
  const facts = host.library.saveFacts(id, photo().facts); await tick();
  expect(ports[0].sent[1]).not.toHaveProperty('edit'); ports[0].confirm(ports[0].sent[1]); await facts;
  ports[0].hold = false; ports[0].confirm(ports[0].sent[0]); await saving;
  expect(ports[0].sidecar).toEqual(edit);
});
it('folds the newest pre-token replacement into the opening snapshot and preserves drop selection', async () => {
  expect(host.library).toBeDefined(); const changed = vi.fn(); host.library.onChange!(changed);
  vi.mocked(bridge.openDropped).mockImplementationOnce(async () => {
    frames('open', 'visit'); frames('watch1', 'visit', 'replace', [{ ...photo(), name: 'old.RAF' }]); frames('watch2', 'visit', 'replace', [{ ...photo(), name: 'latest.RAF' }]); return { token: 'open', selected: [id] };
  });
  setHost(host); await switchFolder(() => host.library.addFiles([new File(['raw'], 'photo.RAF')]));
  expect(useAppStore.getState().files[0].name).toBe('latest.RAF'); expect(useAppStore.getState().selectedFileId).toBe(id); expect(changed).not.toHaveBeenCalled();
});
it('waits for complete frames after the token and rejects a superseded request snapshot', async () => {
  expect(host.library).toBeDefined(); vi.mocked(bridge.openFolder).mockResolvedValueOnce({ token: 'old' }).mockResolvedValueOnce({ token: 'new' });
  const old = host.library.openFolder!(); await tick(); const latest = host.library.openFolder!(); await tick();
  frames('old', 'old'); frames('new', 'new');
  expect(await old).toBeNull(); expect(await latest).toMatchObject({ folder, photos: [{ id }] });
});
it('drops stale listing and facts activations across A to B to A', async () => {
  expect(host.library).toBeDefined(); await host.library.saveFacts(id, photo().facts); const changed = vi.fn(); host.library.onChange!(changed);
  await accept('a1', 'A1'); await accept('b', 'B1', { id: 'B', name: 'B' }); await accept('a2', 'A2');
  frames('stale', 'A1', 'replace'); ports[0].receive({ v: 1, event: 'facts', activation: 'A1', folder, photos: [photo()] });
  expect(changed).not.toHaveBeenCalled();
  frames('current', 'A2', 'replace'); ports[0].receive({ v: 1, event: 'facts', activation: 'A2', folder, photos: [photo()] });
  expect(changed.mock.calls.map(c => c[0].kind)).toEqual(['replace', 'facts']); expect(changed.mock.calls[1][0].snapshot.folder).toEqual(folder);
});
it('awaits flush before replying with the latest shared inventory, and routes menu requests', async () => {
  expect(host.library).toBeDefined(); let finish!: () => void;
  host.library.onFlushRequest!(() => new Promise(resolve => { finish = resolve; }));
  const inventory: UnsavedSummary[] = [{ id, name: 'photo.RAF', folder, error: 'read only' }]; host.library.reportUnsaved!(inventory);
  emit({ kind: 'flush-request', requestId: 7 }); await tick(); expect(bridge.respondFlush).not.toHaveBeenCalled();
  host.library.reportUnsaved!([]); finish(); await tick(); expect(bridge.updateUnsaved).toHaveBeenNthCalledWith(1, inventory); expect(bridge.updateUnsaved).toHaveBeenLastCalledWith([]); expect(bridge.respondFlush).toHaveBeenCalledWith(7, []);
  setHost(host); host.library.onFolderRequest!(f => { void openFolder(f); }); emit({ kind: 'folder-request', folderId: 'A' }); await tick(); expect(bridge.openFolder).toHaveBeenCalledWith('A');
});
it('retries interrupted shared-ledger writes on the new port in revision order', async () => {
  expect(host.library).toBeDefined(); setHost(host); useAppStore.setState({ files: [fromLibraryPhoto(photo())], folder });
  await host.library.saveFacts(id, photo().facts); ports[0].hold = true;
  stop = startPersistence(); host.library.onFlushRequest!(flushPersistence);
  const unsubscribe = onUnsavedChange(edits => host.library.reportUnsaved!(edits.map(({ id, name, folder, error }) => ({ id, name, folder, error }))));
  useAppStore.getState().setFilePreProcessOverride(id, 'exposure', 1); const first = flushPersistence(); await tick();
  expect(ports[0].sent.at(-1)).toMatchObject({ op: 'saveEdit', edit: { preProcessOverrides: { exposure: 1 } } });
  emit({ kind: 'worker-restarted' }); await tick();
  expect(ports).toHaveLength(2); expect(ports[0].closed).toBe(true); expect(ports[1].sent[0]).toMatchObject({ op: 'saveEdit', edit: { preProcessOverrides: { exposure: 1 } } });
  useAppStore.getState().setFilePreProcessOverride(id, 'exposure', 2); await first; await flushPersistence(); await tick();
  expect(ports[1].sent.filter(r => r.op === 'saveEdit').map(r => r.edit.preProcessOverrides.exposure)).toEqual([1, 2]); expect(unsavedEdits()).toEqual([]); expect(bridge.updateUnsaved).toHaveBeenLastCalledWith([]); unsubscribe();
});
it('rejects non-disk drops, reads RAW URLs, rescans on focus and exposes desktop capabilities', async () => {
  expect(host.library).toBeDefined(); vi.mocked(bridge.openDropped).mockRejectedValueOnce(new Error('Only files on disk can be opened.'));
  await expect(host.library.addFiles([new File(['raw'], 'photo.RAF')])).rejects.toThrow('Only files on disk can be opened.'); expect(bridge.openDropped).toHaveBeenCalledWith([expect.any(File)]);
  const bytes = new Uint8Array([1, 2]).buffer; vi.stubGlobal('fetch', vi.fn(async () => ({ arrayBuffer: async () => bytes })));
  expect(await host.library.readRaw(id)).toBe(bytes); expect(fetch).toHaveBeenCalledWith('xveon-photo://raw/' + id);
  win.dispatchEvent(new Event('focus')); await tick(); expect(ports[0].sent.at(-1)).toMatchObject({ op: 'rescan' });
  expect(host.library.remove).toBeUndefined(); expect(host.library.clear).toBeUndefined(); expect(host.library.release).toBeUndefined();
  expect(await host.exporter.status()).toEqual({ available: false, reason: 'Desktop export arrives in M3.' }); expect(await host.display.probe()).toMatchObject({ supported: true, headroom: 2, accurate: false }); expect(host.settingsDbName).toBe('xveon-desktop');
});

it('preserves the current folder without warning on cancelled and superseded drops', async () => {
  expect(host.library).toBeDefined(); setHost(host); useAppStore.setState({ files: [fromLibraryPhoto(photo())], folder }); const warn = vi.spyOn(console, 'warn');
  await importFiles([new File(['x'], 'x.RAF')]); expect(useAppStore.getState().folder).toEqual(folder); expect(warn).not.toHaveBeenCalled();
  let finish!: (value: { token: string } | null) => void;
  vi.mocked(bridge.openDropped).mockImplementationOnce(() => new Promise(resolve => { finish = resolve as typeof finish; }));
  const dropped = importFiles([new File(['x'], 'x.RAF')]); await tick();
  await accept('next', 'next', { id: 'B', name: 'B' }); finish({ token: 'old-drop' }); frames('old-drop', 'old-drop'); await dropped;
  expect(warn).not.toHaveBeenCalled(); expect(useAppStore.getState().folder).toEqual(folder); warn.mockRestore();
});
it('does not post a request to a port replaced between connection and send', async () => {
  await host.library.saveFacts(id, photo().facts); ports[0].sent = [];
  const saving = host.library.saveFacts(id, photo().facts); void saving.catch(() => {});
  emit({ kind: 'worker-restarted' }); await tick();
  try { expect(ports[0].sent).toEqual([]); await expect(saving).rejects.toThrow('Worker connection replaced'); }
  finally { emit({ kind: 'worker-stopped', reason: 'test cleanup' }); }
});
it('delivers current-folder watcher changes while a new folder picker is pending or cancelled', async () => {
  await accept('a', 'A1'); const changed = vi.fn(); host.library.onChange!(changed);
  let cancel!: (value: null) => void;
  vi.mocked(bridge.openFolder).mockImplementationOnce(() => new Promise(resolve => { cancel = resolve; }));
  const watch = listingFrames({ token: 'watch', activation: 'A1', folder, purpose: 'replace' }, [{ ...photo(), name: 'changed.RAF' }]);
  emit({ kind: 'listing', frame: watch[0] }); const opening = host.library.openFolder!();
  for (const frame of watch.slice(1)) emit({ kind: 'listing', frame });
  try { expect(changed).toHaveBeenCalledWith({ kind: 'replace', snapshot: { folder, photos: [{ ...photo(), name: 'changed.RAF' }], complete: true } }); }
  finally { cancel(null); await opening; }
});

it.each(['stale-first', 'current-first'])('cancels a pending handshake and retries the shared ledger on only the current port (%s)', async order => {
  emit({ kind: 'worker-stopped', reason: 'replace initial test host' }); listeners.clear();
  const deliveries: { requestId: string; port: FakePort }[] = [];
  vi.mocked(bridge.requestWorkerPort).mockImplementation(async requestId => { deliveries.push({ requestId, port: new FakePort() }); });
  const deliver = ({ requestId, port }: typeof deliveries[number]) => {
    const event = new MessageEvent('message', { data: { type: 'xveon-port', version: 2, requestId }, origin: 'app://bundle' });
    Object.defineProperties(event, { source: { value: win }, ports: { value: [port] } }); win.dispatchEvent(event);
  };
  host = createDesktopHost(bridge); setHost(host); useAppStore.setState({ files: [fromLibraryPhoto(photo())], folder });
  stop = startPersistence(); host.library.onFlushRequest!(flushPersistence);
  useAppStore.getState().setFilePreProcessOverride(id, 'exposure', 1);
  let settled = false; const interrupted = flushPersistence().then(() => { settled = true; }); await tick();
  emit({ kind: 'worker-restarted' }); await tick(); const cancelledBeforeDelivery = settled;
  const [stale, current] = deliveries;
  if (order === 'stale-first') { deliver(stale); await tick(); deliver(current); }
  else { deliver(current); await tick(); deliver(stale); }
  await tick();
  try {
    expect(cancelledBeforeDelivery).toBe(true);
    expect(current.port.closed).toBe(false); expect(stale.port.sent).toEqual([]);
    // A stale window event queued before restart arrives before the current delivery.
    // Later stale IPC deliveries are closed by preload (covered in its tests).
    if (order === 'stale-first') expect(stale.port.closed).toBe(true);
    expect(current.port.sent.map(r => r.op)).toEqual(['saveEdit', 'saveFacts']);
    expect(current.port.sidecar).toMatchObject({ preProcessOverrides: { exposure: 1 } }); expect(unsavedEdits()).toEqual([]);
    expect(stale.requestId).not.toBe(current.requestId);
  } finally { emit({ kind: 'worker-stopped', reason: 'test cleanup' }); await interrupted; }
});

it('saves the existing shared-ledger edit after three supervisor crashes and explicit Restart', async () => {
  const { EventEmitter } = await import('node:events');
  const { createWorkerSupervisor } = await import('../main/worker');
  class Child extends EventEmitter {
    pid = 42; postMessage() {} kill() { return true; }
  }
  const children: Child[] = [];
  const s = createWorkerSupervisor({ fork: () => { const child = new Child(); children.push(child); queueMicrotask(() => child.emit('spawn')); return child; }, sessionKey: Buffer.from('key'), cacheDir: '/cache', onEvent: event => emit(event === 'restarted' ? { kind: 'worker-restarted' } : { kind: 'worker-stopped', reason: event.stopped }) });
  await s.ready(); await tick();
  let failing = true;
  const connect = bridge.requestWorkerPort;
  bridge.requestWorkerPort = async requestId => { await connect(requestId); ports.at(-1)!.error = failing ? 'worker unavailable' : null; };
  ports[0].error = 'worker unavailable';
  setHost(host); useAppStore.setState({ files: [fromLibraryPhoto(photo())], folder });
  stop = startPersistence(); host.library.onFlushRequest!(flushPersistence);
  useAppStore.getState().setFilePreProcessOverride(id, 'exposure', 2);
  await flushPersistence(); const revision = unsavedEdits()[0].revision;
  for (let i = 0; i < 3; i++) { children[i].emit('exit', 1); await tick(); }
  expect(unsavedEdits()).toMatchObject([{ revision, edit: { preProcessOverrides: { exposure: 2 } } }]);
  expect(ports.at(-1)!.closed).toBe(true);
  failing = false; await s.restart(); await tick();
  expect(ports.at(-1)!.sidecar).toMatchObject({ preProcessOverrides: { exposure: 2 } });
  expect(unsavedEdits()).toEqual([]);
  expect(useAppStore.getState().files[0].editRevision).toBe(revision);
  s.stop();
});

const workerA = '11111111-1111-4111-8111-111111111111';
const workerB = '22222222-2222-4222-8222-222222222222';
function stamped(token: string, revision: number, scan: number, worker = workerA, photos = [photo()]) {
  return listingFrames({ token, activation: 'visit', purpose: 'replace', folder, stamp: { worker, revision, scan } }, photos);
}
it.each(['whole', 'partial'] as const)('rejects an already-posted %s old listing delivered after the actual ledger save acknowledgment', async delivery => {
  await accept('open', 'visit'); setHost(host);
  useAppStore.setState({ files: [fromLibraryPhoto(photo())], folder });
  const changed = vi.fn(); host.library.onChange!(changed);
  stop = startPersistence();
  const delayed = stamped('old', 0, 1);
  if (delivery === 'partial') emit({ kind: 'listing', frame: delayed.shift()! });
  ports[0].hold = true;
  useAppStore.getState().setFilePreProcessOverride(id, 'exposure', 3);
  const saving = flushPersistence(); await tick();
  const editRequest = ports[0].sent.find(r => r.op === 'saveEdit')!;
  ports[0].receive({ v: 1, rid: editRequest.rid, ok: true, stamp: { worker: workerA, revision: 2 } }); await tick();
  const factsRequest = ports[0].sent.find(r => r.op === 'saveFacts')!;
  ports[0].receive({ v: 1, rid: factsRequest.rid, ok: true, stamp: { worker: workerA, revision: 4 } });
  await saving; expect(unsavedEdits()).toEqual([]);
  for (const frame of delayed) emit({ kind: 'listing', frame });
  expect(changed).not.toHaveBeenCalled();
  await tick(); expect(ports[0].sent.at(-1)).toMatchObject({ op: 'rescan' });
  ports[0].confirm(ports[0].sent.at(-1)!); await tick();
  for (const frame of stamped('fresh', 4, 2)) emit({ kind: 'listing', frame });
  expect(changed).toHaveBeenCalledOnce();
});
it.each([false, true])('rejects old scan/worker identities while a picker is pending: %s', async picker => {
  await accept('open', 'visit'); const changed = vi.fn(); host.library.onChange!(changed);
  for (const frame of stamped('new', 0, 2)) emit({ kind: 'listing', frame });
  for (const frame of stamped('old', 0, 1)) emit({ kind: 'listing', frame });
  expect(changed).toHaveBeenCalledOnce();
  emit({ kind: 'worker-restarted', worker: workerB }); await tick();
  let cancel!: (value: null) => void;
  if (picker) vi.mocked(bridge.openFolder).mockImplementationOnce(() => new Promise(resolve => { cancel = resolve; }));
  const opening = picker ? host.library.openFolder!() : Promise.resolve(null);
  for (const frame of stamped('old-worker', 999, 999)) emit({ kind: 'listing', frame });
  for (const frame of stamped('new-worker', 0, 1, workerB)) emit({ kind: 'listing', frame });
  try { expect(changed).toHaveBeenCalledTimes(2); } finally { if (picker) cancel(null); await opening; }
});

it.each(['ordinary', 'second-save', 'cancel', 'empty-drop', 'missing-recent', 'failed-choice', 'startup-cancel', 'startup-empty-drop', 'startup-missing-recent'])('refreshes an opening snapshot across saves and %s', async mode => {
  let finish!: (value: { token: string }) => void;
  vi.mocked(mode.startsWith('startup-') ? bridge.loadLast : bridge.openFolder).mockImplementationOnce(() => new Promise(resolve => { finish = resolve; }));
  const opening = mode.startsWith('startup-') ? host.library.load() : host.library.openFolder!(); await tick();
  for (const frame of listingFrames({ token: 'open', activation: 'visit', purpose: 'open', folder, stamp: { worker: workerA, revision: 0, scan: 1 } }, [photo()])) emit({ kind: 'listing', frame });
  ports[0].hold = true;
  const saving = host.library.save(id, { ...photo().edit, lookPreset: 'umbra' }, photo().facts); await tick();
  ports[0].receive({ v: 1, rid: ports[0].sent[0].rid, ok: true, stamp: { worker: workerA, revision: 2 } }); await tick();
  ports[0].receive({ v: 1, rid: ports[0].sent[1].rid, ok: true, stamp: { worker: workerA, revision: 4 } }); await saving;
  ports[0].hold = false; finish({ token: 'open' }); await tick();
  expect(ports[0].sent.at(-1)).toMatchObject({ op: 'rescan' });
  const secondSave = mode === 'second-save';
  if (mode.includes('empty-drop')) await expect(host.library.addFiles([])).resolves.toBeNull();
  else if (mode.includes('missing-recent')) await expect(host.library.openFolder!({ id: 'missing', name: 'Missing' })).resolves.toBeNull();
  else if (mode === 'cancel' || mode === 'startup-cancel' || mode === 'failed-choice') await expect(host.library.openFolder!()).resolves.toBeNull();
  if (secondSave) {
    ports[0].hold = true;
    const facts = host.library.saveFacts(id, photo().facts); await tick();
    ports[0].receive({ v: 1, rid: ports[0].sent.at(-1)!.rid, ok: true, stamp: { worker: workerA, revision: 6 } }); await facts;
    ports[0].hold = false;
    for (const frame of stamped('stale-again', 4, 2)) emit({ kind: 'listing', frame }); await tick();
    expect(ports[0].sent.filter(request => request.op === 'rescan')).toHaveLength(2);
  }
  for (const frame of stamped('fresh', secondSave ? 6 : 4, 3, workerA, [{ ...photo(), edit: { ...photo().edit, lookPreset: 'umbra' } }])) emit({ kind: 'listing', frame });
  expect((await opening)?.photos[0].edit.lookPreset).toBe('umbra');
});

it.each(['held', 'rejected', 'invalid'])('confirms the edit independently of %s facts', async mode => {
 await host.library.saveFacts(id, photo().facts); const port = ports[0]; port.hold = true;
 const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
 let confirmed = false; const facts = mode === 'invalid' ? { ...photo().facts, cfaType: 'invalid' } as any : photo().facts;
 const saving = host.library.save(id, photo().edit, facts).then(() => { confirmed = true; });
 await tick(); port.confirm(port.sent.at(-1)!); await tick();
 expect(confirmed).toBe(true);
 if (mode === 'rejected') { port.error = 'facts cache full'; port.confirm(port.sent.at(-1)!); await tick(); expect(warn).toHaveBeenCalled(); }
 if (mode === 'held') { port.confirm(port.sent.at(-1)!); await tick(); }
 if (mode === 'invalid') expect(warn).toHaveBeenCalled();
 await saving; warn.mockRestore();
});

it.each(['held', 'rejected', 'invalid'])('clears the shared edit ledger after sidecar ack with %s facts', async mode => {
 setHost(host); useAppStore.setState({ files: [fromLibraryPhoto(photo())], folder });
 await host.library.saveFacts(id, photo().facts); const port = ports[0]; port.hold = true; port.sent = [];
 const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
 if (mode === 'invalid') useAppStore.setState(state => ({ files: state.files.map(file => ({ ...file, cfaType: 'invalid' as any })) }));
 stop = startPersistence(); useAppStore.getState().setFilePreProcessOverride(id, 'exposure', 1); const flushing = flushPersistence(); await tick();
 port.confirm(port.sent.find(r => r.op === 'saveEdit')!); await flushing;
 expect(unsavedEdits()).toEqual([]); expect(useAppStore.getState().files[0].editing).toBe('saved');
 if (mode !== 'invalid') { port.error = mode === 'rejected' ? 'cache denied' : null; port.confirm(port.sent.find(r => r.op === 'saveFacts')!); }
 await tick(); if (mode !== 'held') expect(warn).toHaveBeenCalled(); warn.mockRestore();
});
