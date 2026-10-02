import fs from 'node:fs';
import { EventEmitter } from 'node:events';
import fsp from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { createWorkerController } from './controller';
import { createWatchHandler } from './watch';
import type { WorkerToMain } from '../protocol/rpc';

const nativeWatch = fs.watch;
let events: Array<Record<string, unknown>>;
let evictedEvents: number;
let startedAt: number;
let watchers: fs.FSWatcher[];
function record(kind: string, detail: Record<string, unknown> = {}) {
  if (events.length === 200) { events.shift(); evictedEvents++; }
  events.push({ ms: Date.now() - startedAt, kind, ...detail });
}
beforeEach(() => {
  events = []; evictedEvents = 0; startedAt = Date.now(); watchers = [];
  vi.spyOn(fs, 'watch').mockImplementation(((...args: unknown[]) => {
    record('attach', { path: String(args[0]) });
    const callback = args.at(-1) as (...event: unknown[]) => void;
    try {
      const watcher = (nativeWatch as (...args: unknown[]) => fs.FSWatcher)(...args.slice(0, -1), (...event: unknown[]) => {
        record('callback', { event: event[0], filename: String(event[1]) });
        callback(...event);
      });
      watchers.push(watcher);
      watcher.on('error', error => record('watcher-error', { error: String(error) }));
      return watcher;
    } catch (error) { record('attach-error', { error: String(error) }); throw error; }
  }) as typeof fs.watch);
});

let temp: string | undefined;
const stop: Array<() => void> = [];
afterEach(async () => {
  for (const close of stop.splice(0)) close();
  vi.restoreAllMocks();
  if (temp) await fsp.rm(temp, { recursive: true, force: true });
  temp = undefined;
});

async function setup() {
  temp = await fsp.realpath(await fsp.mkdtemp(path.join(os.tmpdir(), 'xveon-watch-')));
  const folder = path.join(temp, 'photos');
  await fsp.mkdir(folder);
  return folder;
}

function worker(folder: string, debounceMs = 40) {
  const sent: WorkerToMain[] = [];
  let onWatch!: ReturnType<typeof createWatchHandler>;
  const controller = createWorkerController({
    postToMain: message => sent.push(message),
    onWatch: value => onWatch(value),
  });
  onWatch = createWatchHandler(async () => {
    record('replace-start');
    try { await controller.replace(); record('replace-end'); }
    catch (error) { record('replace-error', { error: String(error) }); throw error; }
  }, { debounceMs });
  stop.push(() => onWatch(null));
  const ready = (async () => {
    await controller.handleMain({ v: 1, kind: 'session', key: 'a2V5', cacheDir: path.join(temp!, 'cache') });
    await controller.handleMain({ v: 1, kind: 'roots', realRoots: [folder] });
    await controller.handleMain({ v: 1, kind: 'watch', path: folder, folderId: 'f', activation: 'a' });
  })();
  return { controller, sent, ready, close: () => onWatch(null) };
}

class Port extends EventEmitter {
  peer?: Port;
  postMessage(message: unknown) { this.peer?.emit('message', { data: message }); }
  start() {}
  close() {}
}
function ports() { const a = new Port(), b = new Port(); a.peer = b; b.peer = a; return [a, b]; }

function listings(messages: WorkerToMain[]) {
  const ends = messages.filter(m => m.kind === 'listing-end');
  return ends.map(end => ({
    token: end.token,
    begin: messages.find((m): m is Extract<WorkerToMain, { kind: 'listing-begin' }> => m.kind === 'listing-begin' && m.token === end.token),
    photos: messages.filter((m): m is Extract<WorkerToMain, { kind: 'listing-batch' }> => m.kind === 'listing-batch' && m.token === end.token).flatMap(m => m.photos),
  }));
}

type TestWorker = ReturnType<typeof worker>;
type Listing = ReturnType<typeof listings>[number];
async function until(predicate: () => boolean, w: TestWorker, label: string) {
  const deadline = Date.now() + 3000;
  while (!predicate()) {
    if (Date.now() > deadline) {
      const completed = listings(w.sent);
      throw new Error(`Timed out waiting for ${label}: ${JSON.stringify({
        completedCount: completed.length,
        listings: completed.slice(-20).map(listing => ({
          token: listing.token, begin: listing.begin,
          photos: listing.photos.slice(0, 50).map(photo => ({ name: photo.originalName, exposure: photo.edit.preProcessOverrides.exposure })),
          photoCount: listing.photos.length,
        })),
        events, evictedEvents,
      })}`);
    }
    await new Promise(resolve => setTimeout(resolve, 10));
  }
}
async function replacement(w: TestWorker, label: string, matches: (listing: Listing) => boolean, after = 0) {
  const latest = () => {
    const completed = listings(w.sent), listing = completed.at(-1);
    return completed.length > after && listing?.begin?.purpose === 'replace'
      && listing.begin.activation === 'a' && listing.begin.folder.id === 'f' && matches(listing) ? listing : undefined;
  };
  await until(() => !!latest(), w, label);
  return latest()!;
}
function hasNames(listing: Listing, names: string[]) {
  return JSON.stringify(listing.photos.map(photo => photo.originalName).sort()) === JSON.stringify([...names].sort());
}

it('replaces the listing after a RAW is added and removed', async () => {
  const folder = await setup();
  const w = worker(folder); await w.ready;
  await fsp.writeFile(path.join(folder, 'new.RAF'), 'raw');
  const added = await replacement(w, 'added RAW', listing => hasNames(listing, ['new.RAF']));
  const beforeRemove = listings(w.sent).length;
  await fsp.unlink(path.join(folder, 'new.RAF'));
  const removed = await replacement(w, 'removed RAW', listing => listing.token !== added.token && hasNames(listing, []), beforeRemove);
  expect(removed.token).not.toBe(added.token);
});

it('re-lists a foreign sidecar edit', async () => {
  const folder = await setup();
  await fsp.writeFile(path.join(folder, 'a.RAF'), 'raw');
  const w = worker(folder); await w.ready;
  await fsp.writeFile(path.join(folder, 'a.RAF.xmp'), `<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description rdf:about="" xmlns:xveon="https://naorunaoru.github.io/x-veon/ns/1.0/" xveon:SchemaVersion="1" xveon:Exposure="1.5"/></rdf:RDF></x:xmpmeta>`);
  await replacement(w, 'foreign sidecar exposure', listing => hasNames(listing, ['a.RAF'])
    && listing.photos[0].edit.preProcessOverrides.exposure === 1.5);
});

it('lists all 50 RAW files after native filesystem changes', async () => {
  const folder = await setup();
  const w = worker(folder, 120); await w.ready;
  await Promise.all(Array.from({ length: 50 }, (_, i) => fsp.writeFile(path.join(folder, `${i}.RAF`), 'raw')));
  await replacement(w, 'all 50 added RAWs', listing => hasNames(listing, Array.from({ length: 50 }, (_, i) => `${i}.RAF`)));
});

it('reports a watcher error once and a port rescan still replaces the listing', async () => {
  const folder = await setup();
  const log = vi.spyOn(console, 'error').mockImplementation(() => {});
  const w = worker(folder); await w.ready;
  watchers.at(-1)!.emit('error', new Error('forced watcher error'));
  watchers.at(-1)!.emit('error', new Error('second watcher error'));
  expect(log).toHaveBeenCalledTimes(1);
  expect(String(log.mock.calls[0][0])).toContain(folder);
  expect(String(log.mock.calls[0][0])).toContain('forced watcher error');
  await fsp.writeFile(path.join(folder, 'after.RAF'), 'raw');
  const [a, b] = ports();
  await w.controller.handleMain({ v: 1, kind: 'connect' }, [a]);
  const replies: unknown[] = []; b.on('message', ({ data }) => replies.push(data));
  b.postMessage({ v: 1, rid: 1, op: 'rescan' });
  await replacement(w, 'port rescan after watcher failure', listing => hasNames(listing, ['after.RAF']));
  expect(replies).toEqual([{ v: 1, rid: 1, ok: true }]);
});

it('re-establishes watching when a fresh worker receives watch again', async () => {
  const folder = await setup();
  const first = worker(folder); await first.ready;
  first.close();
  const fresh = worker(folder); await fresh.ready;
  await fsp.writeFile(path.join(folder, 'fresh.RAF'), 'raw');
  await replacement(fresh, 'fresh worker RAW', listing => hasNames(listing, ['fresh.RAF']));
  expect(listings(first.sent)).toHaveLength(0);
});


it('retains the replacement state across notifications separated by quiet intervals', async () => {
  const folder = await setup();
  const source = new EventEmitter();
  vi.spyOn(fs, 'watch').mockImplementationOnce(((...args: unknown[]) => {
    source.on('change', args.at(-1) as (...args: unknown[]) => void);
    return Object.assign(source, { close() { source.removeAllListeners(); } });
  }) as typeof fs.watch);
  const w = worker(folder); await w.ready;
  await fsp.writeFile(path.join(folder, 'new.RAF'), 'raw');
  source.emit('change', 'rename', 'new.RAF');
  const first = await replacement(w, 'first controlled notification', listing => hasNames(listing, ['new.RAF']));
  const beforeSecond = listings(w.sent).length;
  source.emit('change', 'change', 'new.RAF');
  const second = await replacement(w, 'later controlled notification', listing => hasNames(listing, ['new.RAF']), beforeSecond);
  // Both scans completed correctly, with a quiet interval between notifications.
  expect(listings(w.sent)).toHaveLength(2);
  expect(second.token).not.toBe(first.token);
});


it('keeps a late watcher error in bounded timeout diagnostics after a burst', async () => {
  const folder = await setup();
  const log = vi.spyOn(console, 'error').mockImplementation(() => {});
  const w = worker(folder); await w.ready;
  const watcher = watchers.at(-1)!;
  for (let i = 0; i < 205; i++) watcher.emit('change', 'rename', `${i}.RAF`);
  watcher.emit('error', new Error('late watcher failure'));
  const error = await replacement(w, 'RAW after burst', listing => hasNames(listing, ['new.RAF'])).catch(error => error);
  expect(error).toBeInstanceOf(Error);
  const diagnostic = JSON.parse(error.message.slice(error.message.indexOf('{')));
  expect(diagnostic.completedCount).toBe(0);
  expect(diagnostic.events).toHaveLength(200);
  expect(diagnostic.evictedEvents).toBeGreaterThan(0);
  expect(diagnostic.events).toContainEqual(expect.objectContaining({ kind: 'watcher-error', error: 'Error: late watcher failure' }));
  expect(log).toHaveBeenCalledTimes(1);
});
