import type { FolderRef, LibrarySnapshot } from '@/host';
import { useAppStore } from '@/app/store';
import { fromLibraryPhoto, mergeLibraryPhoto } from '@/app/store/photo';
import { matchLens } from '@/app/lens/lensfun';
import { discardResult } from './processing';
import { getHost } from './host';
import {
  cancelPhotoSave, pausePersistence, resumePersistence, flushPersistence, restoreFromLedger, retryUnsaved,
} from './persistence';
let clearing = false;
let suspended = false;
let folderGeneration = 0, issued = 0, accepted = 0;
let dispatch = Promise.resolve();
/** Lets startup reject a restore overtaken by an accepted folder choice. */
export function folderSwitchVersion(): number { return folderGeneration; }
/** Every folder entry point shares this ordering and stale-result guard. */
export async function switchFolder(load: () => Promise<LibrarySnapshot | null>): Promise<void> {
  if (clearing || suspended) return;
  const order = ++issued;
  // Only order flush → host invocation. Picker/list completion stays concurrent,
  // so a later valid choice can supersede an earlier unresolved picker.
  const prepared = dispatch.then(async () => {
    await flushPersistence();
    if (order < accepted || clearing || suspended) return null;
    return { loading: load() };
  });
  dispatch = prepared.then(() => {}, () => {});
  const pending = await prepared;
  if (!pending) return;
  const snapshot = await pending.loading;
  if (!snapshot) return;
  if (order < accepted || clearing || suspended) {
    getHost().library.release?.(snapshot.photos);
    return;
  }
  accepted = order; ++folderGeneration;
  const state = useAppStore.getState();
  const files = snapshot.photos.map(photo => restoreFromLedger(fromLibraryPhoto(photo)));
  for (const file of state.files) discardResult(file.id);
  useAppStore.setState({
    files, folder: snapshot.folder ?? null,
    selectedFileId: snapshot.selectedIds?.[0] ?? files[0]?.id ?? null,
    // The processing service clears its busy gate when the discarded run settles.
    hydrationVersion: state.hydrationVersion + 1,
  });
  for (const photo of snapshot.photos) matchLensFor(photo.id);
}
export async function openFolder(folder?: FolderRef): Promise<void> {
  const library = getHost().library;
  if (library.openFolder) await switchFolder(() => library.openFolder!(folder));
}
export async function importFiles(files: File[]): Promise<void> {
  if (clearing || suspended) return;
  try {
    if (getHost().library.openFolder) {
      await switchFolder(() => getHost().library.addFiles(files));
      return;
    }
    const snapshot = await getHost().library.addFiles(files);
    if (!snapshot || clearing || suspended) return;
    const state = useAppStore.getState();
    state.addFiles(
      snapshot.photos.filter((p) => !state.files.some((f) => f.id === p.id)).map(fromLibraryPhoto),
    );
    if (snapshot.selectedIds?.[0]) useAppStore.getState().selectFile(snapshot.selectedIds[0]);
    for (const photo of snapshot.photos) matchLensFor(photo.id);
  } catch (error) {
    console.warn('RAW import failed:', error);
  }
}
export function startLibraryWatching(): () => void {
  return (
    getHost().library.onChange?.(({ snapshot, kind }) => {
      if (clearing || suspended) return;
      if (snapshot.folder?.id !== useAppStore.getState().folder?.id) return;
      if (kind === 'facts') {
        useAppStore.setState((state) => ({
          files: state.files.map((file) => {
            const photo = snapshot.photos.find((p) => p.id === file.id);
            if (!photo) return file;
            return {
              ...file,
              thumbnailUrl: photo.thumbnailUrl,
              metadata: photo.facts.metadata,
              ...(photo.editing === 'session'
                ? { editing: photo.editing, editingNote: photo.editingNote }
                : {}),
            };
          }),
        }));
      } else {
        const state = useAppStore.getState();
        const live = new Map(state.files.map(file => [file.id, file]));
        const files = snapshot.photos.map(photo => {
          const incoming = restoreFromLedger(fromLibraryPhoto(photo));
          const previous = live.get(photo.id); live.delete(photo.id);
          if (!previous) return incoming;
          const merged = mergeLibraryPhoto(previous, incoming, state);
          if (merged.invalidate) discardResult(photo.id);
          return merged.file;
        });
        for (const id of live.keys()) discardResult(id);
        const selection = snapshot.selectedIds?.[0] ?? state.selectedFileId;
        useAppStore.setState({ files, hydrationVersion: state.hydrationVersion + 1,
          selectedFileId: files.some(f => f.id === selection) ? selection : files[0]?.id ?? null });
        retryUnsaved();
      }
      for (const photo of snapshot.photos) matchLensFor(photo.id);
    }) ?? (() => {})
  );
}
export function matchLensFor(fileId: string): void {
  const file = useAppStore.getState().files.find((f) => f.id === fileId);
  if (!file || file.lensProfile || !file.metadata?.lensModel) return;
  void matchLens(file.metadata.camera, file.metadata.lensModel)
    .then((profile) => {
      if (profile && !clearing) useAppStore.getState().setFileLensProfile(fileId, profile);
    })
    .catch((error) => console.warn('Lens match failed:', error));
}
export async function removeFile(id: string): Promise<void> {
  const remove = getHost().library.remove;
  if (!remove || clearing || suspended) return;
  discardResult(id);
  useAppStore.getState().removeFile(id);
  await cancelPhotoSave(id);
  await remove(id).catch((error) => console.warn('Library removal failed:', error));
}
export async function clearLibrary(): Promise<void> {
  const clear = getHost().library.clear;
  if (!clear || clearing) return;
  clearing = true;
  ++folderGeneration; accepted = ++issued;
  try {
    await pausePersistence();
    for (const file of useAppStore.getState().files) discardResult(file.id);
    await clear();
    useAppStore.setState({ files: [], selectedFileId: null });
    suspended = false;
    resumePersistence();
  } catch (error) {
    // A blocked IDB delete cannot be cancelled. Reopening here would stall behind it.
    // Keep writers/imports suspended until Clear is retried or the app is reloaded.
    suspended = true;
    useAppStore.setState({ files: [], selectedFileId: null });
    throw error;
  } finally {
    clearing = false;
  }
}
