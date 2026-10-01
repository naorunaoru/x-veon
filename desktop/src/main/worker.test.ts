import type { MessagePortMain } from 'electron';
import { EventEmitter } from 'node:events';
import { expect, it, vi } from 'vitest';
import { createWorkerSupervisor } from './worker';
import { isMainToWorker } from '../protocol/rpc';
class Child extends EventEmitter {
  pid: number | undefined;
  sent: any[] = [];
  postMessage(data: any, ports?: any[]) { this.sent.push({ data, ports }); }
  kill() { this.emit('exit', 0); return true; }
  spawn() { this.pid = 42; this.emit('spawn'); }
}
function harness(now = () => 0) {
  const children: Child[] = []; const events: any[] = [];
  const supervisor = createWorkerSupervisor({ fork: () => { const child = new Child(); children.push(child); return child; }, sessionKey: Buffer.from('key'), cacheDir: '/cache', onEvent: e => events.push(e), now, createChannel: () => ({ port1: { close: vi.fn() } as unknown as MessagePortMain, port2: { close: vi.fn() } as unknown as MessagePortMain }) });
  return { supervisor, children, events };
}
it('waits for spawn and restores registry in bounded chunks, roots, replacement and watch after a crash', async () => {
  const { supervisor: s, children, events } = harness();
  for (let i = 0; i < 2001; i++) s.registry.set(String(i).padStart(22, '0'), `/photos/${i}.RAF`);
  s.roots.push('/photos'); s.commitCurrent({ path: '/photos', folderId: 'folder', activation: 'accepted' });
  const ready = s.ready(); expect(children).toHaveLength(1); expect(children[0].sent).toEqual([]); expect(children).toHaveLength(1); children[0].spawn(); await ready;
  const pending = s.request({ kind: 'thumbnail', id: '0'.repeat(22) });
  const rejected = expect(pending).rejects.toThrow('The background worker stopped.');
  await Promise.resolve(); children[0].emit('exit', 1); await rejected;
  expect(children).toHaveLength(2); expect(events).toEqual([]); children[1].spawn(); await s.ready();
  const messages = children[1].sent.map(m => m.data);
  expect(messages.every(isMainToWorker)).toBe(true);
  expect(messages.filter(m => m.kind === 'register').map(m => m.entries.length)).toEqual([1000, 1000, 1]);
  expect(messages[0]).toMatchObject({ kind: 'session', key: Buffer.from('key').toString('base64'), cacheDir: '/cache' });
  expect(messages).toContainEqual({ v: 1, kind: 'roots', realRoots: ['/photos'] });
  expect(messages).toContainEqual(expect.objectContaining({ kind: 'list', path: '/photos', folderId: 'folder', purpose: 'replace', activation: 'accepted' }));
  expect(messages).toContainEqual({ v: 1, kind: 'watch', path: '/photos', folderId: 'folder', activation: 'accepted' });
  expect(events).toEqual(['restarted']); s.stop();
});
it('stops on the third crash within sixty seconds and rejects further work', async () => {
  let now = 0; const { supervisor: s, children, events } = harness(() => now);
  const ready = s.ready(); expect(children).toHaveLength(1); children[0].spawn(); await ready;
  for (let i = 0; i < 3; i++) { now += 10_000; children[i].emit('exit', 1); if (i < 2) { children[i + 1].spawn(); await s.ready(); } }
  expect(children).toHaveLength(3); expect(events).toEqual(['restarted', 'restarted', { stopped: expect.any(String) }]);
  await expect(s.ready()).rejects.toThrow('The background worker stopped.');
});
it('does not count old crashes and does not restart a deliberate quit', async () => {
  let now = 0; const { supervisor: s, children, events } = harness(() => now);
  const ready = s.ready(); expect(children).toHaveLength(1); children[0].spawn(); await ready;
  for (let i = 0; i < 3; i++) { now += 61_000; children[i].emit('exit', 1); children[i + 1].spawn(); await s.ready(); }
  s.stop(); expect(children).toHaveLength(4); expect(events).toEqual(['restarted', 'restarted', 'restarted']);
});
it('settles requests only from validated responses and complete listings', async () => {
  const { supervisor: s, children } = harness(); const ready = s.ready(); expect(children).toHaveLength(1); children[0].spawn(); await ready;
  const result = s.request({ kind: 'list', path: '/photos', folderId: 'f', token: 't', activation: 't', purpose: 'open' });
  await Promise.resolve(); const complete = vi.fn(); void result.then(complete);
  children[0].emit('message', { v: 1, kind: 'listing-begin', token: 't', activation: 't', folder: { id: 'f', name: 'photos' }, total: 0, purpose: 'open' });
  await Promise.resolve(); expect(complete).not.toHaveBeenCalled();
  children[0].emit('message', { v: 2, kind: 'listing-end', token: 't', total: 0 });
  await Promise.resolve(); expect(complete).not.toHaveBeenCalled();
  children[0].emit('message', { v: 1, kind: 'listing-end', token: 't', total: 0 });
  await expect(result).resolves.toMatchObject({ photos: [], folder: { id: 'f', name: 'photos' } });
  const broken = s.request({ kind: 'list', path: '/photos', folderId: 'f', token: 'u', activation: 'u', purpose: 'open' });
  await Promise.resolve(); children[0].emit('message', { v: 1, kind: 'listing-begin', token: 'u', activation: 'u', folder: { id: 'f', name: 'photos' }, total: 1, purpose: 'open' });
  children[0].emit('message', { v: 1, kind: 'listing-end', token: 'u', total: 1 });
  await expect(broken).rejects.toThrow(/total/i); s.stop();
});
it('connects exactly one fresh port on demand and never pushes a replacement after restart', async () => {
  const { supervisor: s, children } = harness(); const delivered = vi.fn(); const connected = s.connect(delivered);
  expect(children).toHaveLength(1); children[0].spawn(); await connected;
  expect(delivered).toHaveBeenCalledOnce(); expect(children[0].sent.find(m => m.data.kind === 'connect').ports).toHaveLength(1);
  children[0].emit('exit', 1); children[1].spawn(); await s.ready();
  expect(delivered).toHaveBeenCalledOnce(); expect(children[1].sent.some(m => m.data.kind === 'connect')).toBe(false); s.stop();
});
it('rejects a request if the worker crashes between readiness and sending', async () => {
  const { supervisor: s, children } = harness(); const ready = s.ready(); children[0].spawn(); await ready;
  const pending = s.request({ kind: 'thumbnail', id: '0'.repeat(22) });
  children[0].emit('exit', 1);
  const result = pending.then(() => 'resolved', error => error.message);
  await Promise.resolve(); await Promise.resolve();
  expect(children[1].sent).toEqual([]);
  await expect(result).resolves.toBe('The background worker stopped.');
  children[1].spawn(); await s.ready(); s.stop();
});
it('rejects pending work and ignores late spawn when quit starts before spawn', async () => {
  const { supervisor: s, children, events } = harness(); const ready = s.ready(); s.stop(); children[0].spawn();
  await expect(ready).rejects.toThrow('The background worker stopped.');
  expect(children).toHaveLength(1); expect(children[0].sent).toEqual([]); expect(events).toEqual([]);
});
it.each(['request', 'send', 'connect'] as const)('rejects %s with the stopped contract when its worker exits before spawn', async operation => {
  const { supervisor: s, children } = harness();
  const pending = operation === 'request' ? s.request({ kind: 'thumbnail', id: '0'.repeat(22) })
    : operation === 'send' ? s.send({ v: 1, kind: 'roots', realRoots: ['/photos'] })
    : s.connect(vi.fn());
  const result = pending.then(() => 'resolved', error => error.message);
  children[0].emit('exit', 1);
  children[1].spawn(); await s.ready();
  const error = await result; s.stop();
  expect(error).toBe('The background worker stopped.');
});
