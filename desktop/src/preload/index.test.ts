import { afterEach, expect, it, vi } from 'vitest';
const m = vi.hoisted(() => ({ exposed: undefined as any, handlers: new Map<string, Function>(), invoke: vi.fn(async () => null), send: vi.fn(), post: vi.fn(), remove: vi.fn(), path: vi.fn(() => '') }));
vi.mock('electron', () => ({ contextBridge: { exposeInMainWorld: (_key: string, value: unknown) => { m.exposed = value; } }, ipcRenderer: { on: (key: string, fn: Function) => { m.handlers.set(key, fn); }, removeListener: m.remove, invoke: m.invoke, send: m.send }, webUtils: { getPathForFile: m.path } }));
afterEach(() => { vi.unstubAllGlobals(); vi.clearAllMocks(); m.handlers.clear(); vi.resetModules(); });
it('exposes only the version-2 API and sends typed channel envelopes', async () => {
  vi.stubGlobal('window', { postMessage: m.post }); vi.stubGlobal('location', { origin: 'app://bundle' }); await import('./index');
  expect(m.exposed.version).toBe(2);
  expect(Object.keys(m.exposed).sort()).toEqual(['chooseExportDestination', 'revealExport', 'version', 'loadLast', 'openFolder', 'openDropped', 'recentFolders', 'displayReadings', 'requestWorkerPort', 'updateUnsaved', 'respondFlush', 'onEvent'].sort());
  await m.exposed.loadLast(); await m.exposed.openFolder('f'); m.path.mockReturnValueOnce('/raw'); await m.exposed.openDropped([new File(['x'], 'x.RAF')]); m.invoke.mockResolvedValueOnce([] as any); await m.exposed.recentFolders(); await m.exposed.requestWorkerPort('00000000-0000-4000-8000-000000000001');
  expect(m.invoke.mock.calls).toEqual([['xveon-desktop', { version: 2, kind: 'loadLast' }], ['xveon-desktop', { version: 2, kind: 'openFolder', folderId: 'f' }], ['xveon-desktop', { version: 2, kind: 'openDropped', paths: ['/raw'] }], ['xveon-desktop', { version: 2, kind: 'recentFolders' }], ['xveon-desktop', { version: 2, kind: 'requestWorkerPort', requestId: '00000000-0000-4000-8000-000000000001' }]]);
  m.exposed.updateUnsaved([]); m.exposed.respondFlush(4, []); expect(m.send.mock.calls).toEqual([['xveon-unsaved', { version: 2, edits: [] }], ['xveon-flush', { version: 2, requestId: 4, unsaved: [] }]]);
});
it('sanitises display responses and rejects invalid readings', async () => {
  await import('./index');
  m.invoke.mockResolvedValueOnce({ potentialEdr: 16, junk: 1 } as any);
  await expect(m.exposed.displayReadings()).resolves.toEqual({ potentialEdr: 16 });
  m.invoke.mockResolvedValueOnce(null);
  await expect(m.exposed.displayReadings()).resolves.toBeNull();
  m.invoke.mockResolvedValueOnce({ potentialEdr: 'x' } as any);
  await expect(m.exposed.displayReadings()).rejects.toThrow('Invalid bridge response');
});
it('filters invalid event and port envelopes and unsubscribes precisely', async () => {
  vi.stubGlobal('window', { postMessage: m.post }); vi.stubGlobal('location', { origin: 'app://bundle' }); await import('./index');
  expect(m.exposed.onEvent).toBeTypeOf('function'); const listener = vi.fn(), unsubscribe = m.exposed.onEvent(listener); const handler = m.handlers.get('xveon-event')!;
  handler({}, { version: 1, kind: 'worker-restarted' }); handler({}, { version: 2, kind: 'flush-request', requestId: 'bad' }); handler({}, { version: 2, kind: 'worker-restarted' });
  expect(listener.mock.calls).toEqual([[{ version: 2, kind: 'worker-restarted' }]]); unsubscribe(); expect(m.remove).toHaveBeenCalledWith('xveon-event', handler);
  await m.exposed.requestWorkerPort('00000000-0000-4000-8000-000000000001'); const port = { close: vi.fn() }; m.handlers.get('xveon-port')!({ ports: [port] }, { version: 1 }); expect(m.post).not.toHaveBeenCalled(); m.handlers.get('xveon-port')!({ ports: [port] }, { version: 2, requestId: '00000000-0000-4000-8000-000000000001' }); expect(m.post).toHaveBeenCalledWith({ type: 'xveon-port', version: 2, requestId: '00000000-0000-4000-8000-000000000001' }, 'app://bundle', [port]);
});
it('rejects malformed or oversized invoke results before exposing them to the renderer', async () => {
  vi.stubGlobal('window', { postMessage: m.post }); vi.stubGlobal('location', { origin: 'app://bundle' }); await import('./index');
  m.invoke.mockResolvedValueOnce({ token: 7 } as any); await expect(m.exposed.openFolder()).rejects.toThrow('Invalid bridge response');
  m.invoke.mockResolvedValueOnce({ token: 't', selected: ['bad'] } as any); m.path.mockReturnValueOnce('/raw'); await expect(m.exposed.openDropped([new File(['x'], 'x.RAF')])).rejects.toThrow('Invalid bridge response');
  m.invoke.mockResolvedValueOnce([{ id: 'f', name: 'x'.repeat(1_000_000) }] as any); await expect(m.exposed.recentFolders()).rejects.toThrow('Invalid bridge response');
});

it.each(['stale-first', 'current-first'])('closes stale deliveries without forwarding or consuming the current handshake (%s)', async order => {
  vi.stubGlobal('window', { postMessage: m.post }); vi.stubGlobal('location', { origin: 'app://bundle' }); await import('./index');
  const old = '00000000-0000-4000-8000-000000000001', current = '00000000-0000-4000-8000-000000000002';
  await m.exposed.requestWorkerPort(old); await m.exposed.requestWorkerPort(current);
  const stalePort = { close: vi.fn() }, currentPort = { close: vi.fn() };
  const receive = m.handlers.get('xveon-port')!;
  const stale = () => receive({ ports: [stalePort] }, { version: 2, requestId: old });
  const fresh = () => receive({ ports: [currentPort] }, { version: 2, requestId: current });
  if (order === 'stale-first') { stale(); fresh(); } else { fresh(); stale(); }
  expect(m.post.mock.calls).toEqual([[{ type: 'xveon-port', version: 2, requestId: current }, 'app://bundle', [currentPort]]]);
  expect(stalePort.close).toHaveBeenCalledOnce(); expect(currentPort.close).not.toHaveBeenCalled();
});

it('rejects forged path strings and empty File paths before invoking main', async () => {
 await import('./index');
 await expect(m.exposed.openDropped(['/private/arbitrary.RAF'])).rejects.toThrow(); expect(m.invoke).not.toHaveBeenCalled();
 m.path.mockReturnValueOnce(''); await expect(m.exposed.openDropped([new File(['x'], 'x.RAF')])).rejects.toThrow(/disk/); expect(m.invoke).not.toHaveBeenCalled();
 expect(m.exposed.pathsForFiles).toBeUndefined();
});

it('validates export destination responses and sends reveal requests', async () => {
 await import('./index'); const id = 'a'.repeat(22), token = '00000000-0000-4000-8000-000000000001';
 m.invoke.mockResolvedValueOnce(null); await expect(m.exposed.chooseExportDestination(id, 'avif')).resolves.toBeNull();
 m.invoke.mockResolvedValueOnce({ token: 'not-a-uuid', name: 'a.avif' } as any); await expect(m.exposed.chooseExportDestination(id, 'avif')).rejects.toThrow('Invalid bridge response');
 m.invoke.mockResolvedValueOnce({ token, name: 'a'.repeat(300) } as any); await expect(m.exposed.chooseExportDestination(id, 'avif')).rejects.toThrow('Invalid bridge response');
 m.invoke.mockResolvedValueOnce({ token, name: 'a.avif' } as any); await expect(m.exposed.chooseExportDestination(id, 'avif')).resolves.toEqual({ token, name: 'a.avif' });
 await m.exposed.revealExport(token); expect(m.invoke).toHaveBeenLastCalledWith('xveon-desktop', { version: 2, kind: 'revealExport', token });
});
