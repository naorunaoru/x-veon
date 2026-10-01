/** The session ledger owns unsaved edits; hosts confirm saves only once durable. */
import { useAppStore, type QueuedFile, type RestoredSettings } from '@/app/store';
import { factsOf, fromLibraryPhoto } from '@/app/store/photo';
import type { FolderRef, PhotoEdit, PhotoFacts, PhotoId } from '@/host';
import { getHost } from './host';
import { getSetting, putSetting, pauseSettings, resumeSettings } from '@/app/storage/settings-storage';
const SETTING_KEYS = [
  'demosaicMethod', 'modelSize', 'exportFormat', 'exportQuality', 'selectedFileId',
] as const;
export interface UnsavedEdit {
  id: PhotoId;
  name: string;
  folder: FolderRef | null;
  revision: number;
  edit: PhotoEdit;
  facts: PhotoFacts;
  deferred: boolean;
  error: string | null;
}
const ledger = new Map<PhotoId, UnsavedEdit>();
const listeners = new Set<(edits: UnsavedEdit[]) => void>();
export function unsavedEdits(): UnsavedEdit[] {
  return [...ledger.values()];
}
export function onUnsavedChange(listener: (edits: UnsavedEdit[]) => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
export function isPhotoDirty(id: PhotoId): boolean {
  return ledger.has(id);
}
function publish() {
  for (const listener of listeners) listener(unsavedEdits());
}
export function restoreFromLedger(file: QueuedFile, entry = ledger.get(file.id)): QueuedFile {
  return entry ? {
    ...file,
    edit: entry.edit,
    editRevision: entry.revision,
    modelNeedsResolution: entry.deferred,
    editing: 'session',
    editingNote: entry.error,
  } : file;
}
interface Writer {
  cancel(id: string): Promise<void>;
  pause(): Promise<void>;
  resume(): void;
  flush(): Promise<void>;
  retry(): void;
}
const writers = new Set<Writer>();
// Survives a subscription remount, so a new writer cannot overlap an older save.
const running = new Map<string, Promise<void>>();
export async function cancelPhotoSave(id: string): Promise<void> {
  if (ledger.delete(id)) publish();
  await Promise.all([...writers].map(w => w.cancel(id)));
}
export async function pausePersistence(): Promise<void> {
  ledger.clear();
  publish();
  await Promise.all([pauseSettings(), ...[...writers].map(w => w.pause())]);
}
export function resumePersistence(): void {
  resumeSettings();
  for (const w of writers) w.resume();
}
export async function flushPersistence(): Promise<void> {
  await Promise.allSettled([...writers].map(w => w.flush()));
  await Promise.allSettled([...running.values()]);
}
export function retryUnsaved(): void {
  for (const w of writers) w.retry();
}
/** Save outcomes belong to the session, even if their originating subscription has stopped. */
function reconcileSave(id: string, revision: number, editing: 'saved' | 'session', editingNote: string | null) {
  useAppStore.setState(state => ({
    files: state.files.map(file =>
      file.id === id && file.editRevision === revision && file.editing !== 'view-only'
        ? { ...file, editing, editingNote } : file,
    ),
  }));
}
export function startPersistence(): () => void {
  let previous = useAppStore.getState();
  let active = true;
  let paused = false;
  const queuedEdits = new Set(ledger.keys());
  const facts = new Map<string, PhotoFacts>();
  const timers = new Map<string, ReturnType<typeof setTimeout>>();
  function clearTimer(id: string) {
    const timer = timers.get(id);
    if (timer !== undefined) clearTimeout(timer);
    timers.delete(id);
  }
  function schedule(id: string, delay = 300) {
    clearTimer(id);
    if (!active || paused) return;
    timers.set(id, setTimeout(() => {
      timers.delete(id);
      void flush(id);
    }, delay));
  }
  async function flush(id: string, force = false): Promise<void> {
    if (!active || paused) return;
    if (running.has(id)) return running.get(id);
    const entry = ledger.get(id);
    const file = useAppStore.getState().files.find(f => f.id === id);
    const item = entry && (force || queuedEdits.has(id)) && !entry.deferred
      && (force || file?.status !== 'processing') ? entry : undefined;
    const fact = facts.get(id);
    if (!item && !fact) return;
    if (item) queuedEdits.delete(id);
    // Consume only this facts payload; newer facts are queued behind this operation.
    if (fact) facts.delete(id);
    const work = (async () => {
      if (item) {
        try {
          await getHost().library.save(id, item.edit, item.facts);
          if (ledger.get(id)?.revision === item.revision) {
            ledger.delete(id);
            reconcileSave(id, item.revision, 'saved', null);
            publish();
          }
        } catch (error) {
          const current = ledger.get(id);
          if (current) {
            const message = error instanceof Error ? error.message : String(error);
            ledger.set(id, { ...current, error: message });
            reconcileSave(id, current.revision, 'session', message);
            publish();
          }
        }
      }
      if (fact) {
        try {
          await getHost().library.saveFacts(id, fact);
        } catch {
          // Facts are a cache; failure never changes the edit's durable state.
        }
      }
    })();
    running.set(id, work);
    await work;
    if (running.get(id) === work) running.delete(id);
    const current = ledger.get(id);
    const processing = useAppStore.getState().files.find(f => f.id === id)?.status === 'processing';
    if (facts.has(id) || (current && queuedEdits.has(id) && !current.deferred && !processing))
      schedule(id, 0);
  }
  const unsubscribe = useAppStore.subscribe(state => {
    const before = previous;
    previous = state;
    if (!active || paused || state.hydrationVersion !== before.hydrationVersion) return;
    for (const key of SETTING_KEYS)
      if (state[key] !== before[key]) void putSetting(key, state[key]).catch(() => {});
    for (const file of state.files) {
      const old = before.files.find(f => f.id === file.id);
      if (!old || old === file) continue;
      const current = ledger.get(file.id);
      const photoFacts = factsOf(file);
      if (file.editing !== 'view-only' && old.editRevision !== file.editRevision) {
        ledger.set(file.id, {
          id: file.id,
          name: file.name,
          folder: state.folder ?? null,
          revision: file.editRevision,
          edit: file.edit,
          facts: photoFacts,
          deferred: file.modelNeedsResolution,
          error: current?.error ?? null,
        });
        publish();
        queuedEdits.add(file.id);
        schedule(file.id);
      } else if (current && file.editing !== 'view-only') {
        const resolved = current.deferred && !file.modelNeedsResolution;
        ledger.set(file.id, {
          ...current, facts: photoFacts,
          ...(resolved ? { edit: file.edit, deferred: false } : {}),
        });
        if (resolved) publish();
        if (resolved || (old.status === 'processing' && file.status !== 'processing')) {
          queuedEdits.add(file.id);
          schedule(file.id);
        }
      }
      if (file.status !== 'processing' && JSON.stringify(factsOf(old)) !== JSON.stringify(photoFacts)) {
        facts.set(file.id, photoFacts);
        schedule(file.id);
      }
    }
  });
  const writer: Writer = {
    cancel: async id => {
      clearTimer(id);
      queuedEdits.delete(id);
      facts.delete(id);
      await running.get(id);
    },
    pause: async () => {
      paused = true;
      for (const id of timers.keys()) clearTimer(id);
      facts.clear();
      queuedEdits.clear();
      await Promise.allSettled([...running.values()]);
    },
    resume: () => {
      previous = useAppStore.getState();
      paused = false;
    },
    retry: () => {
      for (const id of ledger.keys()) {
        queuedEdits.add(id);
        schedule(id, 0);
      }
    },
    flush: async () => {
      const ids = new Set([...ledger.keys(), ...facts.keys(), ...running.keys()]);
      await Promise.allSettled([...ids].map(async id => {
        while (active && !paused) {
          clearTimer(id);
          const revision = ledger.get(id)?.revision;
          await flush(id, true);
          const current = ledger.get(id);
          // Drain newer work, but do not loop on an unchanged failed revision.
          const newerEdit = current && !current.deferred
            && (current.revision !== revision || queuedEdits.has(id));
          if (!facts.has(id) && !newerEdit) break;
        }
      }));
    },
  };
  writers.add(writer);
  window.addEventListener('focus', retryUnsaved);
  return () => {
    active = false;
    unsubscribe();
    window.removeEventListener('focus', retryUnsaved);
    for (const id of timers.keys()) clearTimer(id);
    writers.delete(writer);
  };
}
export async function restore(): Promise<{
  files: QueuedFile[];
  settings: RestoredSettings;
  complete: boolean;
  folder?: FolderRef | null;
}> {
  const [snapshot, ...values] = await Promise.all([
    getHost().library.load(),
    ...SETTING_KEYS.map(key => getSetting(key).catch(() => undefined)),
  ]);
  const settings = Object.fromEntries(SETTING_KEYS.map((key, i) => [key, values[i]])) as RestoredSettings;
  return {
    files: snapshot.photos.map(p => restoreFromLedger(fromLibraryPhoto(p))),
    settings,
    complete: snapshot.complete,
    folder: snapshot.folder,
  };
}
