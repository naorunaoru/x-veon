import { afterEach, expect, it, vi } from 'vitest';
import type { IpcMain, IpcMainInvokeEvent, MessagePortMain } from 'electron';
import { registerDesktopIpc } from './desktop-ipc';
function harness() {
  const handles = new Map<string, (...args: any[]) => any>(), listeners = new Map<string, (...args: any[]) => any>();
  const events: any[] = [], calls: unknown[] = [], postMessage = vi.fn(); const port = {} as MessagePortMain;
  const event = { senderFrame: { postMessage }, trusted: true } as unknown as IpcMainInvokeEvent;
  const guard = registerDesktopIpc({ ipc: { handle: (name: string, handler: (...args: any[]) => any) => { handles.set(name, handler); }, on: (name: string, handler: (...args: any[]) => any) => { listeners.set(name, handler); } } as unknown as IpcMain,
    trusted: e => (e as any).trusted === true, folders: { loadLast: async () => ({ token: 'last' }), openFolder: async id => { calls.push(id); return { token: 'open' }; }, openDropped: async paths => { calls.push(paths); return { token: 'drop', selected: [] }; } }, recent: () => [{ id: 'f', name: 'Folder' }], connect: async deliver => { deliver(port); }, send: e => events.push(e), timeoutMs: 20 });
  return { handles, listeners, events, calls, event, guard, postMessage, port };
}
const edits = [{ id: 'a'.repeat(22), name: 'Photo', folder: { id: 'f', name: 'Folder' }, error: 'locked' }];
afterEach(() => vi.useRealTimers());
it('routes only trusted valid desktop requests, including delivered worker ports', async () => {
  const h = harness(); expect(h.handles.has('xveon-desktop')).toBe(true); const invoke = h.handles.get('xveon-desktop')!;
  await expect(invoke(h.event, { version: 2, kind: 'loadLast' })).resolves.toEqual({ token: 'last' });
  await invoke(h.event, { version: 2, kind: 'openFolder', folderId: 'f' }); await invoke(h.event, { version: 2, kind: 'openDropped', paths: ['/a'] }); expect(h.calls).toEqual(['f', ['/a']]);
  await expect(invoke(h.event, { version: 2, kind: 'recentFolders' })).resolves.toEqual([{ id: 'f', name: 'Folder' }]);
  await invoke(h.event, { version: 2, kind: 'requestWorkerPort' }); expect(h.postMessage).toHaveBeenCalledWith('xveon-port', { version: 2 }, [h.port]);
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
