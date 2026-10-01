import type { FolderRef, LibraryChange, LibraryHost, LibrarySnapshot, UnsavedSummary } from '@/host';
import type { DesktopBridge } from '../protocol/bridge';
import { createListingAssembler, type ListingFrame } from '../protocol/listing';
import { isListingFrame, type ListingStamp } from '../protocol/rpc';
import { photoUrl } from '../protocol/photo-url';
import { createWorkerClient } from './port';
type Completed = { activation: string; snapshot: LibrarySnapshot; stamp?: ListingStamp };
export function createLibrary(bridge: DesktopBridge): LibraryHost {
  const listeners = new Set<(change: LibraryChange) => void>();
  const folderListeners = new Set<(folder?: FolderRef) => void>();
  let flush: (() => Promise<void>) | undefined;
  // This is the ledger's last report, never an independent edit inventory.
  let inventory: UnsavedSummary[] = [];
  let activation: string | undefined, generation = 0, opening = false;
  let worker: string | undefined, acknowledged = 0;
  const retiredWorkers = new Set<string>();
  const scanOrders = new Map<string, number>();
  function fresh(stamp?: ListingStamp): boolean {
    return stamp ? !retiredWorkers.has(stamp.worker) && (worker === undefined || stamp.worker === worker) && stamp.revision >= acknowledged : worker === undefined;
  }
  let wake: (() => void) | undefined;
  const assemblies = new Map<string, { activation: string; assembler: ReturnType<typeof createListingAssembler> }>();
  const opens = new Map<string, Completed | Error>(), replacements = new Map<string, Completed>();
  const publish = (change: LibraryChange) => { for (const listener of listeners) listener(change); };
  const client = createWorkerClient(bridge, event => {
    if (event.activation === activation) publish({ kind: 'facts', snapshot: { folder: event.folder, photos: event.photos, complete: false } });
  }, stamp => {
    if (retiredWorkers.has(stamp.worker) || (worker !== undefined && worker !== stamp.worker)) return;
    worker = stamp.worker; acknowledged = Math.max(acknowledged, stamp.revision); wake?.();
  });
  let rescanRunning = false, rescanNeeded = false;
  function refreshStale(stamp: ListingStamp | undefined, visit: string) {
    if (!stamp || stamp.worker !== worker || stamp.revision >= acknowledged || visit !== activation || opening) return;
    rescanNeeded = true;
    if (rescanRunning) return;
    rescanRunning = true;
    void (async () => {
      try { while (rescanNeeded) { rescanNeeded = false; await client.request({ op: 'rescan' }); } }
      catch { /* Focus/restart will retry if the worker is unavailable. */ }
      finally { rescanRunning = false; }
    })();
  }
  function receive(frame: ListingFrame) {
    if (!isListingFrame(frame)) return;
    if (frame.kind === 'listing-begin') {
      if (assemblies.has(frame.token)) return;
      const { token, activation: visit, stamp } = frame;
      if (stamp && worker === undefined && !retiredWorkers.has(stamp.worker)) worker = stamp.worker;
      if (frame.purpose === 'replace') {
        if (stamp && (retiredWorkers.has(stamp.worker) || stamp.worker !== worker)) return;
        if (!fresh(stamp)) { refreshStale(stamp, visit); if (!opening) return; }
        if (stamp && stamp.scan < (scanOrders.get(visit) ?? 0)) return;
        if (stamp) scanOrders.set(visit, stamp.scan);
      }
      assemblies.set(token, { activation: visit, assembler: createListingAssembler((folder, photos, purpose) => {
        assemblies.delete(token);
        const complete = { activation: visit, stamp, snapshot: { folder, photos, complete: true } };
        if (purpose === 'open') { if (opening) opens.set(token, complete); }
        else if (!stamp || stamp.scan === scanOrders.get(visit)) {
          if (opening) replacements.set(visit, complete);
          if (fresh(stamp) && visit === activation) publish({ kind: 'replace', snapshot: complete.snapshot });
          else refreshStale(stamp, visit);
        }
        wake?.();
      }, (_token, reason) => { assemblies.delete(token); opens.set(token, new Error(reason)); wake?.(); }) });
    }
    assemblies.get(frame.token)?.assembler.push(frame);
  }
  async function request(invoke: () => Promise<{ token: string; selected?: string[] } | null>): Promise<LibrarySnapshot | null> {
    const attempt = ++generation;
    wake?.(); opening = true; opens.clear(); replacements.clear();
    // A picker does not deactivate the current folder or interrupt its watcher.
    for (const [token, listing] of assemblies) if (listing.activation !== activation) assemblies.delete(token);
    try {
      const result = await invoke();
      if (attempt !== generation || !result) return null;
      while (!opens.has(result.token)) {
        await new Promise<void>(resolve => { wake = resolve; });
        if (attempt !== generation) return null;
      }
      const initial = opens.get(result.token)!;
      if (initial instanceof Error) throw initial;
      let rescanned: Completed | undefined;
      for (;;) {
        const latest = replacements.get(initial.activation) ?? initial;
        if (fresh(latest.stamp)) {
          activation = initial.activation;
          return { ...latest.snapshot, ...(result.selected ? { selectedIds: result.selected } : {}) };
        }
        // Main may have held an opening snapshot while a direct-port save was confirmed.
        if (rescanned !== latest) {
          rescanned = latest;
          void client.request({ op: 'rescan' }).catch(error => { opens.set(result.token, error instanceof Error ? error : new Error(String(error))); wake?.(); });
        }
        await new Promise<void>(resolve => { wake = resolve; });
        if (attempt !== generation) return null;
        const failed = opens.get(result.token); if (failed instanceof Error) throw failed;
      }
    } finally {
      if (attempt === generation) {
        opening = false; wake = undefined;
        opens.clear(); replacements.clear();
      }
    }
  }
  bridge.onEvent(event => {
    switch (event.kind) {
      case 'listing': receive(event.frame); break;
      case 'folder-request':
        for (const listener of folderListeners) listener(event.folderId === undefined ? undefined : { id: event.folderId, name: event.folderId });
        break;
      case 'flush-request':
        void Promise.resolve().then(() => flush?.()).catch(() => {}).then(() => bridge.respondFlush(event.requestId, inventory)); break;
      case 'worker-restarted':
        if (worker) retiredWorkers.add(worker);
        worker = event.worker; acknowledged = 0; scanOrders.clear(); assemblies.clear(); replacements.clear(); wake?.();
        void client.restart().then(() => flush?.()).catch(() => {}); break;
      case 'worker-stopped': client.stop(event.reason); break;
    }
  });
  window.addEventListener('focus', () => { void client.request({ op: 'rescan' }).catch(() => {}); });
  return {
    async load() { return await request(() => bridge.loadLast()) ?? { photos: [], complete: true, folder: null }; },
    openFolder: folder => request(() => bridge.openFolder(folder?.id)),
    async addFiles(files) {
      const paths = bridge.pathsForFiles(files);
      if (paths.length !== files.length || paths.some(path => !path)) throw new Error('Only files on disk can be opened.');
      return request(() => bridge.openDropped(paths));
    },
    recentFolders: () => bridge.recentFolders(),
    async readRaw(id) { return (await fetch(photoUrl('raw', id))).arrayBuffer(); },
    async save(id, edit, facts) { await client.request({ op: 'saveEdit', id, edit }); await client.request({ op: 'saveFacts', id, facts }); },
    saveFacts: (id, facts) => client.request({ op: 'saveFacts', id, facts }),
    reportUnsaved(edits) { inventory = edits; bridge.updateUnsaved(edits); },
    onFlushRequest(handler) { flush = handler; return () => { if (flush === handler) flush = undefined; }; },
    onFolderRequest(listener) { folderListeners.add(listener); return () => folderListeners.delete(listener); },
    onChange(listener) { listeners.add(listener); return () => listeners.delete(listener); },
  };
}
