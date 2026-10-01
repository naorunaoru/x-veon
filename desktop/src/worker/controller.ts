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
  onWatch?: (folder: WatchedFolder | null) => void;
}) {
  let library: WorkerLibrary | undefined;
  let port: WorkerPort | undefined;
  let watched: WatchedFolder | null = null;
  const listings = new Map<string, object>();
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
    const run = {}; listings.set(request.token, run);
    const photos: LibraryPhoto[] = []; const registry: [string, string][] = [];
    try {
      for await (const batch of getLibrary().list(request.path, request.folderId)) {
        if (listings.get(request.token) !== run) return;
        photos.push(...batch.photos); registry.push(...batch.registry);
      }
      if (listings.get(request.token) !== run) return;
      const folder = { id: request.folderId, name: path.basename(request.path) || request.path };
      for (const frame of listingFrames({ token: request.token, activation: request.activation, purpose: request.purpose, folder }, photos, registry)) {
        if (listings.get(request.token) !== run) return;
        post(frame);
        if (frame.kind === 'listing-batch') await new Promise<void>(resolve => setImmediate(resolve));
      }
      if (listings.get(request.token) === run) snapshots.set(request.activation, { folder, photos: new Map(photos.map(photo => [photo.id, photo])) });
    } finally { if (listings.get(request.token) === run) listings.delete(request.token); }
  }
  async function replace() {
    if (!watched) return;
    await list({ ...watched, token: randomUUID(), purpose: 'replace' });
  }
  async function handlePort(target: WorkerPort, message: PortRequest) {
    try {
      if (message.op === 'saveEdit') await getLibrary().saveEdit(message.id, message.edit);
      else if (message.op === 'saveFacts') await getLibrary().saveFacts(message.id, message.facts);
      else await replace();
      reply(target, { v: 1, rid: message.rid, ok: true });
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
            library = (opts.createLibrary ?? createWorkerLibrary)({ sessionKey: Buffer.from(message.key, 'base64'), cacheDir: message.cacheDir }); break;
          case 'connect':
            if (ports.length !== 1) return;
            port?.close(); port = ports[0];
            { const target = port; target.on('message', event => { if (port === target && isPortRequest(event.data)) void handlePort(target, event.data); }); target.start(); }
            break;
          case 'register': getLibrary().register(message.entries); break;
          case 'roots': getLibrary().setRoots(message.realRoots); break;
          case 'list': await list(message); break;
          case 'cancel-list': listings.delete(message.token); break;
          case 'watch':
            watched = message.path === null ? null : { path: message.path, folderId: message.folderId!, activation: message.activation! };
            opts.onWatch?.(watched); break;
          case 'thumbnail': {
            const activation = watched?.activation;
            const result = await getLibrary().thumbnail(message.id);
            post({ v: 1, rid: message.rid, kind: 'thumbnail', path: result.path });
            const snapshot = activation && snapshots.get(activation);
            const photo = snapshot && snapshot.photos.get(message.id);
            if (port && watched?.activation === activation && snapshot && photo && result.facts) {
              const updated = { ...photo, facts: result.facts }; snapshot.photos.set(photo.id, updated);
              reply(port, { v: 1, event: 'facts', activation: activation!, folder: snapshot.folder, photos: [updated] });
            }
            break;
          }
        }
      } catch (error) {
        if ('rid' in message) post({ v: 1, rid: message.rid, kind: 'error', error: reason(error).slice(0, 10_000) });
        else throw error;
      }
    },
  };
}
