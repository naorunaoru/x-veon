import type { createExportService } from './exports';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import type { FolderRef, LibraryPhoto } from '@/host';
import type { WorkerLibrary } from './library';
import { createWorkerLibrary } from './library';
import { listingFrames } from '../protocol/listing';
import { isMainToWorker, isPortRequest, isPortReply, isPortEvent, isWorkerToMain } from '../protocol/rpc';
import type { MainToWorker, PortRequest, PortReply, PortEvent, WorkerToMain } from '../protocol/rpc';

export interface WorkerPort {
  on(event: 'message', listener: (event: { data: unknown }) => void): unknown;
  postMessage(message: unknown): void;
  start(): void;
  close(): void;
}
export type WatchedFolder = { path: string; folderId: string; activation: string };
const reason = (e: unknown) => e instanceof Error ? e.message : String(e);
export function createWorkerController(opts: {
  postToMain: (message: WorkerToMain) => void;
  createLibrary?: (options: { sessionKey: Buffer; cacheDir: string }) => WorkerLibrary;
  exports?: ReturnType<typeof createExportService>;
  onWatch?: (folder: WatchedFolder | null) => void;
}) {
  let library: WorkerLibrary | undefined;
  let port: WorkerPort | undefined;
  let watched: WatchedFolder | null = null;
  const listings = new Map<string, object>();
  const replacements = new Map<string, string>();
  const writes = new Map<Promise<unknown>, string>();
  const folderRevisions = new Map<string, number>();
  const photoFolders = new Map<string, string>();
  const headMetadata = new Map<string, { sourceVersion?: string; metadata: NonNullable<LibraryPhoto['facts']['metadata']> }>();
  const writing = (folder: string) => [...writes].filter(([, target]) => target === folder).map(([promise]) => promise);
  let mutation = 0, scan = 0;
  let worker: string = randomUUID();
  const snapshots = new Map<string, { folder: FolderRef; photos: Map<string, LibraryPhoto> }>();
  function post(message: WorkerToMain) {
    if (!isWorkerToMain(message)) throw new Error('Invalid worker response');
    opts.postToMain(message);
  }
  function reply(target: WorkerPort, message: PortReply | PortEvent) {
    if (!isPortReply(message) && !isPortEvent(message)) throw new Error('Invalid port response');
    target.postMessage(message); // Electron window↔worker ports must never use transfer lists.
  }
  function getLibrary() { if (!library) throw new Error('Worker session is not initialized'); return library; }
  async function list(request: { path: string; folderId: string; token: string; activation: string; purpose: 'open' | 'replace' }) {
    if (request.purpose === 'replace') {
      const previous = replacements.get(request.activation);
      if (previous) listings.delete(previous);
      replacements.set(request.activation, request.token);
    }
    const order = ++scan;
    const run = {}; listings.set(request.token, run);
    try {
      while (listings.get(request.token) === run) {
        await Promise.allSettled(writing(request.path));
        const revision = folderRevisions.get(request.path) ?? 0;
        const photos: LibraryPhoto[] = []; const registry: [string, string][] = [];
        for await (const batch of getLibrary().list(request.path, request.folderId)) {
          if (listings.get(request.token) !== run) return;
          photos.push(...batch.photos); registry.push(...batch.registry);
          for (const [id, file] of batch.registry) photoFolders.set(id, path.dirname(file));
        }
        if (listings.get(request.token) !== run) return;
        // A save may finish while the iterator holds an old sidecar or cached facts.
        if (revision !== (folderRevisions.get(request.path) ?? 0) || writing(request.path).length) continue;
        for (const photo of photos) {
          const head = headMetadata.get(photo.id);
          if (head?.sourceVersion === photo.sourceVersion) photo.facts.metadata ??= head?.metadata ?? null;
        }
        const folder = { id: request.folderId, name: path.basename(request.path) || request.path };
        // Publish one complete snapshot without yielding to another scan or save.
        for (const frame of listingFrames({ token: request.token, activation: request.activation, purpose: request.purpose, folder, stamp: { worker, revision: mutation, scan: order } }, photos, registry)) post(frame);
        snapshots.set(request.activation, { folder, photos: new Map(photos.map(photo => [photo.id, photo])) });
        return;
      }
    } finally { if (listings.get(request.token) === run) listings.delete(request.token); }
  }
  async function replace() {
    if (!watched) return;
    await list({ ...watched, token: randomUUID(), purpose: 'replace' });
  }
  async function mutate<T>(id: string, operation: () => Promise<T>): Promise<T> {
    const folder = photoFolders.get(id) ?? watched?.path ?? '';
    const changed = () => { ++mutation; folderRevisions.set(folder, (folderRevisions.get(folder) ?? 0) + 1); };
    changed();
    const write = Promise.resolve().then(operation); writes.set(write, folder);
    try { return await write; } finally { writes.delete(write); changed(); }
  }
  function getExports() { if (!opts.exports) throw new Error('Export service is unavailable'); return opts.exports; }
  async function handlePort(target: WorkerPort, message: PortRequest) {
    try {
      const rid = message.rid;
      switch (message.op) {
        case 'exportStatus': reply(target, { v: 1, rid, ok: true, availability: getExports().status() }); return;
        case 'exportBegin': await getExports().begin(message); break;
        case 'exportChunk': getExports().chunk(message.job, message.plane, message.offset, message.data); break;
        case 'exportCommit': reply(target, { v: 1, rid, ok: true, receipt: await getExports().commit(message.job) }); return;
        case 'exportCancel': getExports().cancel(message.job); break;
        case 'rescan': await replace(); break;
        case 'saveEdit': await mutate(message.id, () => getLibrary().saveEdit(message.id, message.edit)); break;
        case 'saveFacts': await mutate(message.id, () => getLibrary().saveFacts(message.id, message.facts)); break;
      }
      const saved = message.op === 'saveEdit' || message.op === 'saveFacts';
      reply(target, { v: 1, rid, ok: true, ...(saved ? { stamp: { worker, revision: mutation } } : {}) });
    } catch (error) { reply(target, { v: 1, rid: message.rid, ok: false, error: reason(error).slice(0, 10_000) }); }
  }
  return {
    replace,
    get watched() { return watched; },
    async handleMain(value: unknown, ports: WorkerPort[] = []) {
      if (!isMainToWorker(value)) return;
      const message: MainToWorker = value;
      try {
        switch (message.kind) {
          case 'session':
            if (library) throw new Error('Worker session is already initialized');
            worker = message.worker ?? worker;
            library = (opts.createLibrary ?? createWorkerLibrary)({ sessionKey: Buffer.from(message.key, 'base64'), cacheDir: message.cacheDir }); break;
          case 'connect':
            if (ports.length !== 1) return;
            opts.exports?.cancelAll();
            port?.close(); port = ports[0];
            { const target = port; target.on('message', event => { if (port === target && isPortRequest(event.data)) void handlePort(target, event.data); }); target.start(); }
            break;
          case 'export-destination':
            if (message.rid !== undefined && !path.isAbsolute(message.path)) throw new Error('Invalid export destination');
            getExports().register(message.token, message.path);
            if (message.rid !== undefined) post({ v: 1, rid: message.rid, kind: 'export-destination', token: message.token });
            break;
          case 'register': getLibrary().register(message.entries); for (const [id, file] of message.entries) photoFolders.set(id, path.dirname(file)); break;
          case 'roots': getLibrary().setRoots(message.realRoots); break;
          case 'list': await list(message); break;
          case 'cancel-list': listings.delete(message.token); break;
          case 'watch':
            watched = message.path === null ? null : { path: message.path, folderId: message.folderId!, activation: message.activation! };
            opts.onWatch?.(watched); break;
          case 'thumbnail': {
            const activation = watched?.activation;
            const result = await getLibrary().thumbnail(message.id);
            if (result.facts?.metadata) headMetadata.set(message.id, { sourceVersion: result.sourceVersion, metadata: result.facts.metadata });
            post({ v: 1, rid: message.rid, kind: 'thumbnail', path: result.path });
            const snapshot = activation && snapshots.get(activation);
            const photo = snapshot && snapshot.photos.get(message.id);
            if (port && watched?.activation === activation && snapshot && photo && result.facts && result.sourceVersion === photo.sourceVersion) {
              const updated = { ...photo, facts: result.facts }; snapshot.photos.set(photo.id, updated);
              reply(port, { v: 1, event: 'facts', activation: activation!, folder: snapshot.folder, photos: [updated] });
            }
            break;
          }
        }
      } catch (error) {
        if ('rid' in message && message.rid !== undefined) post({ v: 1, rid: message.rid, kind: 'error', error: reason(error).slice(0, 10_000) });
        else throw error;
      }
    },
  };
}
