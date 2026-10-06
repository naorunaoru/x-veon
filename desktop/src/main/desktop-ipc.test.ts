import { afterEach, expect, it, vi } from 'vitest';
import type { IpcMain, IpcMainInvokeEvent, MessagePortMain } from 'electron';
import { registerDesktopIpc } from './desktop-ipc';
function harness() {
  const handles = new Map<string, (...args: any[]) => any>(), listeners = new Map<string, (...args: any[]) => any>();
  const events: any[] = [], calls: unknown[] = [], postMessage = vi.fn(), readings = vi.fn(() => ({ potentialEdr: 16 })); const port = {} as MessagePortMain;
  const event = { senderFrame: { postMessage }, trusted: true } as unknown as IpcMainInvokeEvent;
  const guard = registerDesktopIpc({ exports: { choose: async () => null, reveal() {} }, ipc: { handle: (name: string, handler: (...args: any[]) => any) => { handles.set(name, handler); }, on: (name: string, handler: (...args: any[]) => any) => { listeners.set(name, handler); } } as unknown as IpcMain,
    trusted: e => (e as any).trusted === true, folders: { loadLast: async () => ({ token: 'last' }), openFolder: async id => { calls.push(id); return { token: 'open' }; }, openDropped: async paths => { calls.push(paths); return { token: 'drop', selected: [] }; } }, display: { readings }, recent: () => [{ id: 'f', name: 'Folder' }], connect: async deliver => { deliver(port); }, send: e => events.push(e), timeoutMs: 20 });
  return { handles, listeners, events, calls, event, guard, postMessage, port, readings };
}
it('routes display readings only for a trusted sender', async () => {
  const h = harness(), invoke = h.handles.get('xveon-desktop')!;
  await expect(invoke(h.event, { version: 2, kind: 'displayReadings' })).resolves.toEqual({ potentialEdr: 16 });
  expect(h.readings).toHaveBeenCalledOnce();
  await expect(invoke({ trusted: false }, { version: 2, kind: 'displayReadings' })).rejects.toThrow('Invalid bridge request');
  expect(h.readings).toHaveBeenCalledOnce();
});
const edits = [{ id: 'a'.repeat(22), name: 'Photo', folder: { id: 'f', name: 'Folder' }, error: 'locked' }];
afterEach(() => vi.useRealTimers());
it('routes only trusted valid desktop requests, including delivered worker ports', async () => {
  const h = harness(); expect(h.handles.has('xveon-desktop')).toBe(true); const invoke = h.handles.get('xveon-desktop')!;
  await expect(invoke(h.event, { version: 2, kind: 'loadLast' })).resolves.toEqual({ token: 'last' });
  await invoke(h.event, { version: 2, kind: 'openFolder', folderId: 'f' }); await invoke(h.event, { version: 2, kind: 'openDropped', paths: ['/a'] }); expect(h.calls).toEqual(['f', ['/a']]);
  await expect(invoke(h.event, { version: 2, kind: 'recentFolders' })).resolves.toEqual([{ id: 'f', name: 'Folder' }]);
  await invoke(h.event, { version: 2, kind: 'requestWorkerPort', requestId: '00000000-0000-4000-8000-000000000001' }); expect(h.postMessage).toHaveBeenCalledWith('xveon-port', { version: 2, requestId: '00000000-0000-4000-8000-000000000001' }, [h.port]);
  await expect(invoke({ trusted: false }, { version: 2, kind: 'loadLast' })).rejects.toThrow('Invalid bridge request'); await expect(invoke(h.event, { version: 2, kind: 'openDropped', paths: [1] })).rejects.toThrow('Invalid bridge request');
});
it('takes inventory only from trusted valid pushes and correlates flush responses', async () => {
  const h = harness(); expect(h.listeners.has('xveon-unsaved')).toBe(true); const update = h.listeners.get('xveon-unsaved')!, reply = h.listeners.get('xveon-flush')!;
  update(h.event, { version: 2, edits }); expect(h.guard.inventory()).toEqual(edits); update({ trusted: false }, { version: 2, edits: [] }); expect(h.guard.inventory()).toEqual(edits);
  const flush = h.guard.requestFlush(); const requestId = h.events.at(-1).requestId; expect(h.events.at(-1).kind).toBe('flush-request');
  reply(h.event, { version: 2, requestId: requestId + 1, unsaved: [] }); expect(h.guard.inventory()).toEqual(edits);
  reply(h.event, { version: 2, requestId, unsaved: [] }); await expect(flush).resolves.toEqual([]); expect(h.guard.inventory()).toEqual([]);
});
it('expires pending flushes to the most recent pushed inventory', async () => {
  vi.useFakeTimers(); const h = harness(); const flush = h.guard.requestFlush(); expect(h.events).toHaveLength(1); h.listeners.get('xveon-unsaved')!(h.event, { version: 2, edits });
  await vi.advanceTimersByTimeAsync(20); await expect(flush).resolves.toEqual(edits);
  h.listeners.get('xveon-flush')!(h.event, { version: 2, requestId: h.events[0].requestId, unsaved: [] }); expect(h.guard.inventory()).toEqual(edits);
});

it('serializes overlapping port connections and closes a late superseded delivery before connecting the current request', async () => {
  let invoke!: (...args: any[]) => Promise<unknown>;
  const connections: { deliver: (port: MessagePortMain) => void; finish: () => void }[] = [];
  registerDesktopIpc({ exports: { choose: async () => null, reveal() {} }, ipc: { handle: (_name: string, handler: typeof invoke) => { invoke = handler; }, on() {} } as unknown as IpcMain,
    trusted: () => true, folders: { loadLast: async () => null, openFolder: async () => null, openDropped: async () => null }, display: { readings: () => null }, recent: () => [], send() {},
    connect: deliver => new Promise<void>(finish => { connections.push({ deliver, finish }); }),
  });
  const postMessage = vi.fn(), event = { senderFrame: { postMessage } };
  const old = invoke(event, { version: 2, kind: 'requestWorkerPort', requestId: '00000000-0000-4000-8000-000000000001' });
  for (let i=0;i<5;i++) await Promise.resolve();
  const current = invoke(event, { version: 2, kind: 'requestWorkerPort', requestId: '00000000-0000-4000-8000-000000000002' });
  for (let i=0;i<5;i++) await Promise.resolve();
  const oldPort = { close: vi.fn() } as unknown as MessagePortMain;
  try {
    expect(connections).toHaveLength(1); connections[0].deliver(oldPort); connections[0].finish(); await old;
    for (let i=0;i<5;i++) await Promise.resolve();
    expect(oldPort.close).toHaveBeenCalledOnce(); expect(postMessage).not.toHaveBeenCalled(); expect(connections).toHaveLength(2);
    const currentPort = { close: vi.fn() } as unknown as MessagePortMain;
    connections[1].deliver(currentPort); connections[1].finish(); await current;
    expect(postMessage).toHaveBeenCalledWith('xveon-port', { version: 2, requestId: '00000000-0000-4000-8000-000000000002' }, [currentPort]);
  } finally { for (const c of connections) c.finish(); }
});
it('continues with the latest queued port request after an older connection rejects', async () => {
  let invoke!: (...args: any[]) => Promise<unknown>;
  let fail!: (error: Error) => void; const port = { close: vi.fn() } as unknown as MessagePortMain;
  const connect = vi.fn<(deliver: (port: MessagePortMain) => void) => Promise<void>>()
    .mockImplementationOnce(() => new Promise((_resolve, reject) => { fail = reject; }))
    .mockImplementationOnce(async deliver => { deliver(port); });
  registerDesktopIpc({ exports: { choose: async () => null, reveal() {} }, ipc: { handle: (_name: string, handler: typeof invoke) => { invoke = handler; }, on() {} } as unknown as IpcMain,
    trusted: () => true, folders: { loadLast: async () => null, openFolder: async () => null, openDropped: async () => null }, display: { readings: () => null }, recent: () => [], send() {}, connect,
  });
  const postMessage = vi.fn(), event = { senderFrame: { postMessage } };
  const request = (n: number) => invoke(event, { version: 2, kind: 'requestWorkerPort', requestId: `00000000-0000-4000-8000-00000000000${n}` });
  const old = request(1); const rejected = expect(old).rejects.toThrow('worker exited');
  for (let i=0;i<5;i++) await Promise.resolve();
  const middle = request(2), latest = request(3); fail(new Error('worker exited'));
  await rejected; await middle; await latest;
  expect(connect).toHaveBeenCalledTimes(2);
  expect(postMessage.mock.calls).toEqual([['xveon-port', { version: 2, requestId: '00000000-0000-4000-8000-000000000003' }, [port]]]);
});

it('routes exports only for trusted requests', async () => {
 let invoke!: (...args: any[]) => Promise<unknown>;
 const token = '00000000-0000-4000-8000-000000000001', photoId = 'a'.repeat(22);
 const exports = { choose: vi.fn(async () => ({ token, name: 'a.avif' })), reveal: vi.fn() };
 registerDesktopIpc({ ipc: { handle: (_name: string, fn: typeof invoke) => { invoke = fn; }, on() {} } as unknown as IpcMain, trusted: e => (e as any).trusted === true,
 folders: { loadLast: async () => null, openFolder: async () => null, openDropped: async () => null }, display: { readings: () => null }, recent: () => [], connect: async () => {}, send() {}, exports });
 await expect(invoke({ trusted: false }, { version: 2, kind: 'chooseExportDestination', photoId, format: 'avif' })).rejects.toThrow(); expect(exports.choose).not.toHaveBeenCalled();
 await expect(invoke({ trusted: true }, { version: 2, kind: 'chooseExportDestination', photoId, format: 'avif' })).resolves.toEqual({ token, name: 'a.avif' });
 expect(exports.choose).toHaveBeenCalledWith(photoId, 'avif');
 await expect(invoke({ trusted: false }, { version: 2, kind: 'revealExport', token })).rejects.toThrow(); expect(exports.reveal).not.toHaveBeenCalled();
 await invoke({ trusted: true }, { version: 2, kind: 'revealExport', token }); expect(exports.reveal).toHaveBeenCalledWith(token);
});
