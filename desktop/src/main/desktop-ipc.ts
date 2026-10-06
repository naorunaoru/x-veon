import { isDesktopRequest, isUnsavedUpdate, isFlushResponse } from '../protocol/security';
import type { IpcMain, IpcMainEvent, IpcMainInvokeEvent, MessagePortMain } from 'electron';
import type { DisplayReadings, FolderRef, NewerRelease, UnsavedSummary } from '@/host';
import type { BridgeEvent } from '../protocol/bridge';
type Deps = { exports: Pick<ReturnType<typeof import('./exports').createExportDestinations>, 'choose' | 'reveal'>; ipc: Pick<IpcMain, 'handle' | 'on'>; trusted(event: IpcMainEvent | IpcMainInvokeEvent): boolean; folders: { loadLast(): Promise<unknown>; openFolder(id?: string): Promise<unknown>; openDropped(paths: string[]): Promise<unknown> }; display: { readings(): DisplayReadings | null }; updates: { check(): Promise<NewerRelease | null> }; recent(): FolderRef[]; connect(deliver: (port: MessagePortMain) => void): Promise<void>; send(event: BridgeEvent): void; timeoutMs?: number };
export function registerDesktopIpc(deps: Deps) {
  let inventory: UnsavedSummary[] = [], nextId = 0;
  let requestedPort: string | undefined;
  let connecting = Promise.resolve();
  const flushes = new Map<number, { resolve(unsaved: UnsavedSummary[]): void; timer: ReturnType<typeof setTimeout> }>();
  deps.ipc.handle('xveon-desktop', async (event, value: unknown) => {
    if (!deps.trusted(event) || !isDesktopRequest(value)) throw new Error('Invalid bridge request');
    switch (value.kind) {
      case 'chooseExportDestination': return deps.exports.choose(value.photoId, value.format);
      case 'revealExport': return deps.exports.reveal(value.token);
      case 'loadLast': return deps.folders.loadLast();
      case 'openFolder': return deps.folders.openFolder(value.folderId);
      case 'openDropped': return deps.folders.openDropped(value.paths);
      case 'recentFolders': return deps.recent();
      case 'displayReadings': return deps.display.readings();
      case 'checkForUpdate': return deps.updates.check();
      case 'requestWorkerPort': {
        requestedPort = value.requestId;
        // The supervisor attaches a port before delivering it. Serialize attempts so
        // an older asynchronous connect cannot detach the newer worker connection.
        const connection = connecting.then(async () => {
          if (requestedPort !== value.requestId) return;
          await deps.connect(port => {
            if (requestedPort !== value.requestId) { port.close(); return; }
            event.senderFrame!.postMessage('xveon-port', { version: 2, requestId: value.requestId }, [port]);
          });
        });
        connecting = connection.catch(() => {});
        return connection;
      }
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
