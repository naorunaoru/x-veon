/** Shared debounce/retry ownership. Hosts resolve saves only once durable. */
import { useAppStore, type QueuedFile, type RestoredSettings } from '@/app/store';
import { factsOf, fromLibraryPhoto } from '@/app/store/photo';
import { getHost } from './host';
import { getSetting, putSetting, pauseSettings, resumeSettings } from '@/app/storage/settings-storage';
const SETTING_KEYS = [
  'demosaicMethod',
  'modelSize',
  'exportFormat',
  'exportQuality',
  'selectedFileId',
] as const;
interface Pending {
  file: QueuedFile;
  revision: number;
}
interface Writer {
  cancel(id: string): Promise<void>;
  pause(): Promise<void>;
  resume(): void;
}
const writers = new Set<Writer>();
export async function cancelPhotoSave(id: string): Promise<void> {
  await Promise.all([...writers].map((w) => w.cancel(id)));
}
export async function pausePersistence(): Promise<void> {
  await Promise.all([pauseSettings(), ...[...writers].map((w) => w.pause())]);
}
export function resumePersistence(): void {
  resumeSettings();
  for (const w of writers) w.resume();
}
export function startPersistence(): () => void {
  let previous = useAppStore.getState();
  let active = true;
  let paused = false;
  let revision = 0;
  const pending = new Map<string, Pending>();
  const timers = new Map<string, ReturnType<typeof setTimeout>>();
  const running = new Map<string, Promise<void>>();
  function clearTimer(id: string) {
    const timer = timers.get(id);
    if (timer) clearTimeout(timer);
    timers.delete(id);
  }
  function stateOf(id: string, editing: 'saved' | 'session', editingNote: string | null) {
    if (!active || paused) return;
    useAppStore.setState((state) => ({
      files: state.files.map((f) =>
        f.id === id && f.editing !== 'view-only' ? { ...f, editing, editingNote } : f,
      ),
    }));
  }
  function schedule(id: string, delay = 300) {
    clearTimer(id);
    if (!active || paused) return;
    timers.set(
      id,
      setTimeout(() => {
        timers.delete(id);
        void flush(id);
      }, delay),
    );
  }
  async function flush(id: string): Promise<void> {
    if (!active || paused || running.has(id)) return;
    const item = pending.get(id);
    if (!item) return;
    const work = (async () => {
      try {
        await getHost().library.save(id, item.file.edit, factsOf(item.file));
        if (pending.get(id) === item) {
          pending.delete(id);
          stateOf(id, 'saved', null);
        }
      } catch (error) {
        if (pending.has(id)) stateOf(id, 'session', error instanceof Error ? error.message : String(error));
      }
    })();
    running.set(id, work);
    await work;
    running.delete(id);
    if (pending.has(id) && pending.get(id) !== item) schedule(id, 0);
  }
  const unsubscribe = useAppStore.subscribe((state) => {
    const before = previous;
    previous = state;
    if (!active || paused || state.hydrationVersion !== before.hydrationVersion) return;
    for (const key of SETTING_KEYS)
      if (state[key] !== before[key]) void putSetting(key, state[key]).catch(() => {});
    for (const file of state.files) {
      const old = before.files.find((f) => f.id === file.id);
      if (!old || old === file || file.editing === 'view-only' || file.status === 'processing') continue;
      if (old.edit === file.edit && JSON.stringify(factsOf(old)) === JSON.stringify(factsOf(file))) continue;
      // Do not write an incomplete neural edit while its first model is still being resolved.
      if (file.status !== 'error' && file.modelNeedsResolution)
        continue;
      pending.set(file.id, { file, revision: ++revision });
      schedule(file.id);
    }
    for (const id of pending.keys())
      if (!state.files.some((f) => f.id === id)) {
        clearTimer(id);
        pending.delete(id);
      }
  });
  const retry = () => {
    for (const id of pending.keys()) schedule(id, 0);
  };
  window.addEventListener('focus', retry);
  const writer: Writer = {
    cancel: async (id) => {
      clearTimer(id);
      pending.delete(id);
      await running.get(id);
    },
    pause: async () => {
      paused = true;
      for (const id of timers.keys()) clearTimer(id);
      pending.clear();
      await Promise.allSettled([...running.values()]);
    },
    resume: () => {
      previous = useAppStore.getState();
      paused = false;
    },
  };
  writers.add(writer);
  return () => {
    active = false;
    unsubscribe();
    window.removeEventListener('focus', retry);
    for (const id of timers.keys()) clearTimer(id);
    pending.clear();
    writers.delete(writer);
  };
}
export async function restore(): Promise<{
  files: QueuedFile[];
  settings: RestoredSettings;
  complete: boolean;
}> {
  const [snapshot, ...values] = await Promise.all([
    getHost().library.load(),
    ...SETTING_KEYS.map((key) => getSetting(key).catch(() => undefined)),
  ]);
  const settings = Object.fromEntries(SETTING_KEYS.map((key, i) => [key, values[i]])) as RestoredSettings;
  return { files: snapshot.photos.map(fromLibraryPhoto), settings, complete: snapshot.complete };
}
