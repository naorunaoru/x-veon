import fs from 'node:fs';
import { EventEmitter } from 'node:events';
import fsp from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { afterEach, expect, it, vi } from 'vitest';
import { createWorkerController } from './controller';
import { createWatchHandler } from './watch';
import type { WorkerToMain } from '../protocol/rpc';

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
  onWatch = createWatchHandler(() => controller.replace(), { debounceMs });
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
    begin: messages.find(m => m.kind === 'listing-begin' && m.token === end.token),
    photos: messages.filter((m): m is Extract<WorkerToMain, { kind: 'listing-batch' }> => m.kind === 'listing-batch' && m.token === end.token).flatMap(m => m.photos),
  }));
}

async function until(predicate: () => boolean, timeoutMs = 3000) {
  const deadline = Date.now() + timeoutMs;
  while (!predicate()) {
    if (Date.now() > deadline) throw new Error('Timed out waiting for watcher listing');
    await new Promise(resolve => setTimeout(resolve, 10));
  }
}

it('replaces the listing after a RAW is added and removed', async () => {
  const folder = await setup();
  const w = worker(folder); await w.ready;
  await fsp.writeFile(path.join(folder, 'new.RAF'), 'raw');
  await until(() => listings(w.sent).length === 1);
  expect(listings(w.sent)[0].photos.map(p => p.originalName)).toEqual(['new.RAF']);
  expect(listings(w.sent)[0].begin).toMatchObject({ purpose: 'replace', activation: 'a' });
  await fsp.unlink(path.join(folder, 'new.RAF'));
  await until(() => listings(w.sent).length === 2);
  expect(listings(w.sent)[1].photos).toEqual([]);
  expect(listings(w.sent)[1].token).not.toBe(listings(w.sent)[0].token);
});

it('re-lists a foreign sidecar edit', async () => {
  const folder = await setup();
  await fsp.writeFile(path.join(folder, 'a.RAF'), 'raw');
  const w = worker(folder); await w.ready;
  await fsp.writeFile(path.join(folder, 'a.RAF.xmp'), `<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description rdf:about="" xmlns:xveon="https://naorunaoru.github.io/x-veon/ns/1.0/" xveon:SchemaVersion="1" xveon:Exposure="1.5"/></rdf:RDF></x:xmpmeta>`);
  await until(() => listings(w.sent).length === 1);
  expect(listings(w.sent)[0].photos[0].edit.preProcessOverrides.exposure).toBe(1.5);
});

it('coalesces 50 changes into one replacement listing', async () => {
  const folder = await setup();
  const w = worker(folder, 120); await w.ready;
  await Promise.all(Array.from({ length: 50 }, (_, i) => fsp.writeFile(path.join(folder, `${i}.RAF`), 'raw')));
  await until(() => listings(w.sent).length >= 1);
  await new Promise(resolve => setTimeout(resolve, 250));
  expect(listings(w.sent)).toHaveLength(1);
  expect(listings(w.sent)[0].photos).toHaveLength(50);
});

it('reports a watcher error once and a port rescan still replaces the listing', async () => {
  const folder = await setup();
  let watcher: fs.FSWatcher | undefined;
  const original = fs.watch;
  vi.spyOn(fs, 'watch').mockImplementation(((...args: Parameters<typeof fs.watch>) => {
    watcher = original(...args);
    return watcher;
  }) as typeof fs.watch);
  const log = vi.spyOn(console, 'error').mockImplementation(() => {});
  const w = worker(folder); await w.ready;
  watcher!.emit('error', new Error('forced watcher error'));
  watcher!.emit('error', new Error('second watcher error'));
  expect(log).toHaveBeenCalledTimes(1);
  expect(String(log.mock.calls[0][0])).toContain(folder);
  expect(String(log.mock.calls[0][0])).toContain('forced watcher error');
  await fsp.writeFile(path.join(folder, 'after.RAF'), 'raw');
  const [a, b] = ports();
  await w.controller.handleMain({ v: 1, kind: 'connect' }, [a]);
  const replies: unknown[] = []; b.on('message', ({ data }) => replies.push(data));
  b.postMessage({ v: 1, rid: 1, op: 'rescan' });
  await until(() => listings(w.sent).length === 1);
  expect(replies).toEqual([{ v: 1, rid: 1, ok: true }]);
  expect(listings(w.sent)).toHaveLength(1);
  expect(listings(w.sent)[0].photos.map(p => p.originalName)).toEqual(['after.RAF']);
});

it('re-establishes watching when a fresh worker receives watch again', async () => {
  const folder = await setup();
  const first = worker(folder); await first.ready;
  first.close();
  const fresh = worker(folder); await fresh.ready;
  await fsp.writeFile(path.join(folder, 'fresh.RAF'), 'raw');
  await until(() => listings(fresh.sent).length === 1);
  expect(listings(fresh.sent)[0].photos.map(p => p.originalName)).toEqual(['fresh.RAF']);
  expect(listings(first.sent)).toHaveLength(0);
});
