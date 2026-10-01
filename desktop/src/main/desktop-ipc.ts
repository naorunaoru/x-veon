import { isDesktopRequest, isUnsavedUpdate, isFlushResponse } from '../protocol/security';
import type { IpcMain, IpcMainEvent, IpcMainInvokeEvent, MessagePortMain } from 'electron';
import type { FolderRef, UnsavedSummary } from '@/host';
import type { BridgeEvent } from '../protocol/bridge';
type Deps = { ipc: Pick<IpcMain, 'handle' | 'on'>; trusted(event: IpcMainEvent | IpcMainInvokeEvent): boolean; folders: { loadLast(): Promise<unknown>; openFolder(id?: string): Promise<unknown>; openDropped(paths: string[]): Promise<unknown> }; recent(): FolderRef[]; connect(deliver: (port: MessagePortMain) => void): Promise<void>; send(event: BridgeEvent): void; timeoutMs?: number };
export function registerDesktopIpc(deps: Deps) {
  let inventory: UnsavedSummary[] = [], nextId = 0;
  const flushes = new Map<number, { resolve(unsaved: UnsavedSummary[]): void; timer: ReturnType<typeof setTimeout> }>();
  deps.ipc.handle('xveon-desktop', async (event, value: unknown) => {
    if (!deps.trusted(event) || !isDesktopRequest(value)) throw new Error('Invalid bridge request');
    switch (value.kind) {
      case 'loadLast': return deps.folders.loadLast();
      case 'openFolder': return deps.folders.openFolder(value.folderId);
      case 'openDropped': return deps.folders.openDropped(value.paths);
      case 'recentFolders': return deps.recent();
      case 'requestWorkerPort': return deps.connect(port => event.senderFrame!.postMessage('xveon-port', { version: 2 }, [port]));
    }
  });
  deps.ipc.on('xveon-unsaved', (event, value: unknown) => { if (deps.trusted(event) && isUnsavedUpdate(value)) inventory = value.edits; });
  deps.ipc.on('xveon-flush', (event, value: unknown) => {
    if (!deps.trusted(event) || !isFlushResponse(value)) return;
    const pending = flushes.get(value.requestId); if (!pending) return;
    clearTimeout(pending.timer); flushes.delete(value.requestId); inventory = value.unsaved; pending.resolve(inventory);
  });
  return {
    inventory: () => inventory,
    requestFlush: () => new Promise<UnsavedSummary[]>(resolve => {
      const requestId = ++nextId;
      const timer = setTimeout(() => { flushes.delete(requestId); resolve(inventory); }, deps.timeoutMs ?? 3000);
      flushes.set(requestId, { resolve, timer });
      try { deps.send({ kind: 'flush-request', requestId }); }
      catch { clearTimeout(timer); flushes.delete(requestId); resolve(inventory); }
    }),
  };
}
