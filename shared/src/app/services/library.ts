import { useAppStore } from '@/app/store';
import { fromLibraryPhoto } from '@/app/store/photo';
import { matchLens } from '@/app/lens/lensfun';
import { discardResult } from './processing';
import { getHost } from './host';
import { cancelPhotoSave, pausePersistence, resumePersistence } from './persistence';
let clearing = false;
let suspended = false;
export async function importFiles(files: File[]): Promise<void> {
  if (clearing || suspended) return;
  try {
    const snapshot = await getHost().library.addFiles(files);
    if (clearing || suspended) return;
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
        state.restoreFromDb(
          snapshot.photos.map((photo) => {
            const local = state.files.find((f) => f.id === photo.id);
            return local?.editing === 'session' || (local?.editRevision ?? 0) > 0
              ? local!
              : fromLibraryPhoto(photo);
          }),
          {},
        );
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
