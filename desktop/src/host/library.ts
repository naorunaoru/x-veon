import type { FolderRef, LibraryChange, LibraryHost, LibrarySnapshot, UnsavedSummary } from '@/host';
import type { DesktopBridge } from '../protocol/bridge';
import { createListingAssembler, type ListingFrame } from '../protocol/listing';
import { isListingFrame } from '../protocol/rpc';
import { photoUrl } from '../protocol/photo-url';
import { createWorkerClient } from './port';
type Completed = { activation: string; snapshot: LibrarySnapshot };
export function createLibrary(bridge: DesktopBridge): LibraryHost {
  const listeners = new Set<(change: LibraryChange) => void>();
  const folderListeners = new Set<(folder?: FolderRef) => void>();
  let flush: (() => Promise<void>) | undefined;
  // This is the ledger's last report, never an independent edit inventory.
  let inventory: UnsavedSummary[] = [];
  let activation: string | undefined, generation = 0, opening = false;
  let wake: (() => void) | undefined;
  const assemblies = new Map<string, { activation: string; assembler: ReturnType<typeof createListingAssembler> }>();
  const opens = new Map<string, Completed | Error>(), replacements = new Map<string, Completed>();
  const publish = (change: LibraryChange) => { for (const listener of listeners) listener(change); };
  const client = createWorkerClient(bridge, event => {
    if (event.activation === activation) publish({ kind: 'facts', snapshot: { folder: event.folder, photos: event.photos, complete: false } });
  });
  function receive(frame: ListingFrame) {
    if (!isListingFrame(frame)) return;
    if (frame.kind === 'listing-begin') {
      if (assemblies.has(frame.token)) return;
      const { token, activation: visit } = frame;
      assemblies.set(token, { activation: visit, assembler: createListingAssembler((folder, photos, purpose) => {
        assemblies.delete(token);
        const complete = { activation: visit, snapshot: { folder, photos, complete: true } };
        if (purpose === 'open') { if (opening) opens.set(token, complete); }
        else if (visit === activation) publish({ kind: 'replace', snapshot: complete.snapshot });
        else if (opening) replacements.set(visit, complete);
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
      activation = initial.activation;
      const latest = replacements.get(activation) ?? initial;
      return { ...latest.snapshot, ...(result.selected ? { selectedIds: result.selected } : {}) };
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
