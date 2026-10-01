import { afterEach, expect, it, vi } from 'vitest';
const m = vi.hoisted(() => ({ exposed: undefined as any, handlers: new Map<string, Function>(), invoke: vi.fn(async () => null), send: vi.fn(), post: vi.fn(), remove: vi.fn(), path: vi.fn(() => '') }));
vi.mock('electron', () => ({ contextBridge: { exposeInMainWorld: (_key: string, value: unknown) => { m.exposed = value; } }, ipcRenderer: { on: (key: string, fn: Function) => { m.handlers.set(key, fn); }, removeListener: m.remove, invoke: m.invoke, send: m.send }, webUtils: { getPathForFile: m.path } }));
afterEach(() => { vi.unstubAllGlobals(); vi.clearAllMocks(); m.handlers.clear(); vi.resetModules(); });
it('exposes only the version-2 API and sends typed channel envelopes', async () => {
  vi.stubGlobal('window', { postMessage: m.post }); vi.stubGlobal('location', { origin: 'app://bundle' }); await import('./index');
  expect(m.exposed.version).toBe(2);
  expect(Object.keys(m.exposed).sort()).toEqual(['version', 'loadLast', 'openFolder', 'openDropped', 'recentFolders', 'pathsForFiles', 'requestWorkerPort', 'updateUnsaved', 'respondFlush', 'onEvent'].sort());
  await m.exposed.loadLast(); await m.exposed.openFolder('f'); await m.exposed.openDropped(['/raw']); m.invoke.mockResolvedValueOnce([] as any); await m.exposed.recentFolders(); await m.exposed.requestWorkerPort();
  expect(m.invoke.mock.calls).toEqual([['xveon-desktop', { version: 2, kind: 'loadLast' }], ['xveon-desktop', { version: 2, kind: 'openFolder', folderId: 'f' }], ['xveon-desktop', { version: 2, kind: 'openDropped', paths: ['/raw'] }], ['xveon-desktop', { version: 2, kind: 'recentFolders' }], ['xveon-desktop', { version: 2, kind: 'requestWorkerPort' }]]);
  m.exposed.updateUnsaved([]); m.exposed.respondFlush(4, []); expect(m.send.mock.calls).toEqual([['xveon-unsaved', { version: 2, edits: [] }], ['xveon-flush', { version: 2, requestId: 4, unsaved: [] }]]);
  expect(m.exposed.pathsForFiles([new File(['x'], 'x')])).toEqual(['']);
});
it('filters invalid event and port envelopes and unsubscribes precisely', async () => {
  vi.stubGlobal('window', { postMessage: m.post }); vi.stubGlobal('location', { origin: 'app://bundle' }); await import('./index');
  expect(m.exposed.onEvent).toBeTypeOf('function'); const listener = vi.fn(), unsubscribe = m.exposed.onEvent(listener); const handler = m.handlers.get('xveon-event')!;
  handler({}, { version: 1, kind: 'worker-restarted' }); handler({}, { version: 2, kind: 'flush-request', requestId: 'bad' }); handler({}, { version: 2, kind: 'worker-restarted' });
  expect(listener.mock.calls).toEqual([[{ version: 2, kind: 'worker-restarted' }]]); unsubscribe(); expect(m.remove).toHaveBeenCalledWith('xveon-event', handler);
  const port = {}; m.handlers.get('xveon-port')!({ ports: [port] }, { version: 1 }); expect(m.post).not.toHaveBeenCalled(); m.handlers.get('xveon-port')!({ ports: [port] }, { version: 2 }); expect(m.post).toHaveBeenCalledWith({ type: 'xveon-port', version: 2 }, 'app://bundle', [port]);
});
it('rejects malformed or oversized invoke results before exposing them to the renderer', async () => {
  vi.stubGlobal('window', { postMessage: m.post }); vi.stubGlobal('location', { origin: 'app://bundle' }); await import('./index');
  m.invoke.mockResolvedValueOnce({ token: 7 } as any); await expect(m.exposed.openFolder()).rejects.toThrow('Invalid bridge response');
  m.invoke.mockResolvedValueOnce({ token: 't', selected: ['bad'] } as any); await expect(m.exposed.openDropped(['/raw'])).rejects.toThrow('Invalid bridge response');
  m.invoke.mockResolvedValueOnce([{ id: 'f', name: 'x'.repeat(1_000_000) }] as any); await expect(m.exposed.recentFolders()).rejects.toThrow('Invalid bridge response');
});
