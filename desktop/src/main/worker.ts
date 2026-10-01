import { randomUUID } from 'node:crypto';
import type { EventEmitter } from 'node:events';
import type { MessagePortMain } from 'electron';
import type { FolderRef, LibraryPhoto, PhotoId } from '@/host';
import { waitForWorkerSpawn } from './worker-ready';
import { createListingAssembler } from '../protocol/listing';
import { isMainToWorker, isWorkerToMain, isListingFrame } from '../protocol/rpc';
import type { MainToWorker, WorkerToMain, ListingStamp } from '../protocol/rpc';

export type UtilityProcessLike = EventEmitter & {
  readonly pid?: number;
  postMessage(message: unknown, ports?: MessagePortMain[]): void;
  kill(): boolean;
};
type CommittedFolder = { path: string; folderId: string; activation: string };
type ListRequest = Omit<Extract<MainToWorker, { kind: 'list' }>, 'v' | 'rid'>;
type ThumbnailRequest = Omit<Extract<MainToWorker, { kind: 'thumbnail' }>, 'v' | 'rid'>;
export type ListingResult = { folder: FolderRef; photos: LibraryPhoto[]; purpose: 'open' | 'replace'; token: string; activation: string; registry: [PhotoId, string][]; stamp?: ListingStamp };
type Pending = { reject(error: Error): void; resolve(value: ListingResult | Extract<WorkerToMain, { kind: 'thumbnail' }>): void; token?: string; assembler?: ReturnType<typeof createListingAssembler>; entries: [PhotoId, string][] };
const stoppedMessage = 'The background worker stopped.';

export function createWorkerSupervisor(opts: {
  fork: () => UtilityProcessLike;
  sessionKey: Buffer;
  cacheDir: string;
  onEvent: (event: 'restarted' | { stopped: string }) => void;
  now?: () => number;
  createChannel?: () => { port1: MessagePortMain; port2: MessagePortMain } | Promise<{ port1: MessagePortMain; port2: MessagePortMain }>;
}) {
  const registry = new Map<PhotoId, string>();
  const roots: string[] = [];
  const pending = new Map<number, Pending>();
  const listeners = new Set<(message: WorkerToMain) => void>();
  let current: CommittedFolder | null = null;
  let child: UtilityProcessLike | undefined;
  let readiness: Promise<void> | undefined;
  let instance = randomUUID();
  let stopped = false;
  let quitting = false;
  let nextRid = 1;
  let crashes: number[] = [];
  const now = opts.now ?? Date.now;
  function rejectPending() { for (const request of pending.values()) request.reject(new Error(stoppedMessage)); pending.clear(); }
  function post(target: UtilityProcessLike, message: MainToWorker, ports?: MessagePortMain[]) {
    if (!isMainToWorker(message)) throw new Error('Invalid worker message');
    target.postMessage(message, ports);
  }
  function received(target: UtilityProcessLike, value: unknown) {
    if (child !== target || stopped || !isWorkerToMain(value)) return;
    if (isListingFrame(value)) {
      const entry = [...pending.values()].find(p => p.token === value.token);
      if (entry) {
        if (value.kind === 'listing-batch' && value.registry) entry.entries.push(...value.registry);
        entry.assembler!.push(value);
      }
    } else {
      const entry = pending.get(value.rid);
      if (entry) {
        if (value.kind === 'error') { pending.delete(value.rid); entry.reject(new Error(value.error)); }
        else if (!entry.token) { pending.delete(value.rid); entry.resolve(value); }
      }
    }
    for (const listener of listeners) listener(value);
  }
  function start(restarting = false): Promise<void> {
    if (stopped) return Promise.reject(new Error(stoppedMessage));
    const target = opts.fork(); child = target; instance = randomUUID();
    target.on('message', value => received(target, value));
    target.once('exit', () => {
      if (child !== target || stopped) return;
      child = undefined; readiness = undefined; rejectPending();
      const time = now(); crashes = crashes.filter(t => time - t < 60_000); crashes.push(time);
      if (crashes.length >= 3) { stopped = true; opts.onEvent({ stopped: stoppedMessage }); }
      else { void start(true).catch(() => {}); }
    });
    readiness = waitForWorkerSpawn(target).then(() => {
      if (stopped || child !== target) throw new Error(stoppedMessage);
      post(target, { v: 1, kind: 'session', key: opts.sessionKey.toString('base64'), cacheDir: opts.cacheDir, worker: instance });
      let entries: [PhotoId, string][] = [];
      for (const entry of registry) {
        const candidate = { v: 1 as const, kind: 'register' as const, entries: [...entries, entry] };
        if (!isMainToWorker(candidate) && entries.length) { post(target, { v: 1, kind: 'register', entries }); entries = []; }
        entries.push(entry);
      }
      if (entries.length) post(target, { v: 1, kind: 'register', entries });
      post(target, { v: 1, kind: 'roots', realRoots: [...roots] });
      if (current) {
        post(target, { v: 1, rid: nextRid++, kind: 'list', ...current, token: `restore-${nextRid}`, purpose: 'replace' });
        post(target, { v: 1, kind: 'watch', ...current });
      }
      if (restarting) opts.onEvent('restarted');
    }).catch(error => {
      if (child === target && !stopped) {
        stopped = true; rejectPending(); target.kill(); opts.onEvent({ stopped: String(error) });
      }
      throw error;
    });
    return readiness;
  }
  function ready(): Promise<void> { return stopped ? Promise.reject(new Error(stoppedMessage)) : readiness ?? start(); }
  async function send(message: MainToWorker): Promise<void> {
    const initialized = ready(), target = child;
    try { await initialized; }
    catch { throw new Error(stoppedMessage); }
    if (stopped || child !== target) throw new Error(stoppedMessage);
    if (message.kind === 'cancel-list') {
      for (const [rid, entry] of pending) if (entry.token === message.token) { pending.delete(rid); entry.assembler?.cancel(message.token); entry.reject(new Error('Listing cancelled')); }
    }
    post(target!, message);
  }
  function request(message: ListRequest): Promise<ListingResult>;
  function request(message: ThumbnailRequest): Promise<Extract<WorkerToMain, { kind: 'thumbnail' }>>;
  async function request(message: ListRequest | ThumbnailRequest): Promise<ListingResult | Extract<WorkerToMain, { kind: 'thumbnail' }>> {
    const initialized = ready(), target = child;
    try { await initialized; }
    catch { throw new Error(stoppedMessage); }
    if (stopped || child !== target) throw new Error(stoppedMessage);
    const rid = nextRid++;
    return new Promise((resolve, reject) => {
      const entry: Pending = { resolve, reject, entries: [] };
      if (message.kind === 'list') {
        entry.token = message.token;
        entry.assembler = createListingAssembler((folder, photos, purpose, stamp) => {
          pending.delete(rid); resolve({ folder, photos, purpose, token: message.token, activation: message.activation, registry: entry.entries, stamp });
        }, (_token, reason) => { pending.delete(rid); reject(new Error(reason)); });
      }
      pending.set(rid, entry);
      try { post(target!, { v: 1, rid, ...message }); }
      catch (error) { pending.delete(rid); reject(error); }
    });
  }
  return {
    registry, roots, ready, request, send,
    async restart(): Promise<void> {
      if (quitting) throw new Error(stoppedMessage);
      if (!stopped) return ready();
      stopped = false; crashes = []; readiness = undefined;
      await start(true);
    },
    async prepareCommit(): Promise<(folder: CommittedFolder, entries: [PhotoId, string][], remember: () => void) => void> {
      const initialized = ready(), target = child;
      try { await initialized; } catch { throw new Error(stoppedMessage); }
      return (folder, entries, remember) => {
        if (stopped || child !== target) throw new Error(stoppedMessage);
        const nextRoots = roots.includes(folder.path) ? [...roots] : [...roots, folder.path];
        const rootMessage: MainToWorker = { v: 1, kind: 'roots', realRoots: nextRoots };
        const watchMessage: MainToWorker = { v: 1, kind: 'watch', ...folder };
        if (!isMainToWorker(rootMessage) || !isMainToWorker(watchMessage)) throw new Error('Invalid folder commit');
        const registrations: MainToWorker[] = [];
        let batch: [PhotoId, string][] = [];
        for (const entry of entries) {
          if (batch.length && !isMainToWorker({ v: 1, kind: 'register', entries: [...batch, entry] })) {
            registrations.push({ v: 1, kind: 'register', entries: batch }); batch = [];
          }
          batch.push(entry);
          if (!isMainToWorker({ v: 1, kind: 'register', entries: batch })) throw new Error('Invalid folder registry');
        }
        if (batch.length) registrations.push({ v: 1, kind: 'register', entries: batch });
        remember();
        current = { ...folder };
        for (const [id, file] of entries) registry.set(id, file);
        roots.splice(0, roots.length, ...nextRoots);
        for (const message of registrations) post(target!, message);
        post(target!, rootMessage); post(target!, watchMessage);
      };
    },
    get current() { return current; },
    get instance() { return instance; },
    // Main calls this only once it accepts a folder request (Task 9).
    commitCurrent(folder: CommittedFolder | null) { current = folder && { ...folder }; },
    onMessage(listener: (message: WorkerToMain) => void) { listeners.add(listener); return () => { listeners.delete(listener); }; },
    async connect(postToWindow: (port: MessagePortMain) => void) {
      const initialized = ready(), target = child;
      try { await initialized; }
      catch { throw new Error(stoppedMessage); }
      if (stopped || child !== target) throw new Error(stoppedMessage);
      const channel = await (opts.createChannel?.() ?? import('electron').then(({ MessageChannelMain }) => new MessageChannelMain()));
      try {
        if (stopped || child !== target) throw new Error(stoppedMessage);
        post(target!, { v: 1, kind: 'connect' }, [channel.port1]); postToWindow(channel.port2);
      } catch (error) { channel.port1.close(); channel.port2.close(); throw error; }
    },
    stop() {
      quitting = true;
      if (stopped) return;
      stopped = true; rejectPending(); const target = child;
      if (target?.pid !== undefined) target.kill();
      else if (target) void waitForWorkerSpawn(target).then(() => target.kill(), () => {});
    },
  };
}
