import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { realpath, stat } from 'node:fs/promises';
import { RAW_EXTENSIONS } from '@/lib/catalog';
import type { PhotoId } from '@/host';
import { folderId, isMissingFolder, type createFolderStore } from './folders';
import type { createWorkerSupervisor } from './worker';
import { createListingAssembler, bridgeListingFrames, type ListingFrame } from '../protocol/listing';
import { isListingFrame } from '../protocol/rpc';
type Deps = { store: ReturnType<typeof createFolderStore>; worker: ReturnType<typeof createWorkerSupervisor>; chooseFolder(): Promise<string | null>; send(frame: ListingFrame): void; error(folder: string, message: string): void; accepted(): void };
type Choice = { path: string; selected?: string[] } | null;
type Result = { token: string; selected?: PhotoId[] } | null;
export function createFolderRequests(deps: Deps) {
  let issued = 0, accepted = 0;
  const active = new Map<number, { token: string; listing: boolean; cancel(): void }>();
  let publication: { activation: string; opened: boolean; replacement?: ListingFrame[] } | undefined;
  const replacements = new Map<string, { activation: string; entries: [PhotoId, string][]; assembler: ReturnType<typeof createListingAssembler> }>();
  deps.worker.onMessage(message => {
    if (!isListingFrame(message)) return;
    if (message.kind === 'listing-begin') {
      if (message.purpose !== 'replace' || message.activation !== deps.worker.current?.activation) return;
      const header = message;
      const replacement = { activation: message.activation, entries: [] as [PhotoId, string][], assembler: createListingAssembler((folder, photos, purpose) => {
        replacements.delete(header.token);
        if (deps.worker.current?.activation !== header.activation) return;
        for (const [id, file] of replacement.entries) deps.worker.registry.set(id, file);
        try {
          const frames = bridgeListingFrames({ folder, purpose, token: header.token, activation: header.activation, stamp: header.stamp }, photos);
          if (publication?.activation === header.activation && !publication.opened) publication.replacement = frames;
          else for (const frame of frames) deps.send(frame);
        } catch (error) { deps.error(header.folder.name, error instanceof Error ? error.message : String(error)); }
      }, (_token, reason) => { replacements.delete(header.token); deps.error(header.folder.name, reason); }) };
      replacements.set(message.token, replacement);
    }
    const replacement = replacements.get(message.token);
    if (!replacement) return;
    if (replacement.activation !== deps.worker.current?.activation) { replacements.delete(message.token); return; }
    if (message.kind === 'listing-batch' && message.registry) replacement.entries.push(...message.registry);
    replacement.assembler.push(message);
  });
  function start(choose: () => Promise<Choice>, quiet = false): Promise<Result> {
    const order = ++issued;
    const token = randomUUID();
    let cancel!: () => void;
    const cancelled = new Promise<null>(resolve => { cancel = () => { active.delete(order); resolve(null); }; });
    const request = { token, cancel, listing: false };
    active.set(order, request);
    const latest = () => order >= accepted;
    const run = (async (): Promise<Result> => {
      let folderPath = '';
      try {
        const choice = await choose(); if (!choice || !latest()) return null;
        folderPath = choice.path;
        const canonical = await realpath(folderPath);
        if (!(await stat(canonical)).isDirectory()) throw new Error('The selected path is not a folder');
        if (!latest()) return null;
        accepted = order;
        // Release even requests still waiting for the picker or an OS path lookup.
        for (const [older, previous] of active) if (older < order) {
          previous.cancel();
          if (previous.listing) void deps.worker.send({ v: 1, kind: 'cancel-list', token: previous.token }).catch(() => {});
        }
        request.listing = true;
        const id = folderId(canonical);
        const listing = await deps.worker.request({ kind: 'list', path: canonical, folderId: id, token, activation: token, purpose: 'open' });
        if (!latest()) return null;
        const frames = bridgeListingFrames({ token, activation: token, folder: listing.folder, purpose: 'open', stamp: listing.stamp }, listing.photos);
        const commit = await deps.worker.prepareCommit(); if (!latest()) return null;
        commit({ path: canonical, folderId: id, activation: token }, listing.registry, () => deps.store.remember(canonical));
        publication = { activation: token, opened: false };
        deps.accepted();
        for (const frame of frames) deps.send(frame);
        publication.opened = true;
        for (const frame of publication.replacement ?? []) deps.send(frame);
        publication.replacement = undefined;
        void deps.store.flush().catch(error => deps.error(path.basename(canonical) || canonical, error instanceof Error ? error.message : String(error)));
        const selected = choice.selected && listing.registry.filter(([, file]) => choice.selected!.includes(file)).map(([id]) => id);
        return { token, ...(selected ? { selected } : {}) };
      } catch (error) {
        if (latest()) {
          if (isMissingFolder(error) && deps.store.forgetPath(folderPath)) deps.accepted();
          if (!quiet) deps.error(path.basename(folderPath) || folderPath, error instanceof Error ? error.message : String(error));
        }
        return null;
      } finally { active.delete(order); }
    })();
    return Promise.race([run, cancelled]);
  }
  return {
    loadLast: () => start(async () => { const last = deps.store.last(); const folder = last && deps.store.resolve(last.id); return folder ? { path: folder } : null; }, true),
    openFolder: (id?: string) => start(async () => { const folder = id === undefined ? await deps.chooseFolder() : deps.store.resolve(id); return folder ? { path: folder } : null; }),
    openDropped: (paths: string[]) => start(async () => {
      const existing: { path: string; directory: boolean }[] = [];
      for (const file of paths) {
        try { const canonical = await realpath(file); const info = await stat(canonical); if (info.isDirectory() || (info.isFile() && RAW_EXTENSIONS.includes(path.extname(canonical).toLowerCase()))) existing.push({ path: canonical, directory: info.isDirectory() }); } catch { /* Skip missing/unreadable drops. */ }
      }
      if (!existing.length) return null;
      const first = existing[0];
      return { path: first.directory ? first.path : path.dirname(first.path), selected: existing.filter(f => !f.directory).map(f => f.path) };
    }),
  };
}
