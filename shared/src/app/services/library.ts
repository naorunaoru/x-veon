import { useAppStore } from '@/app/store';
import type { QueuedFile } from '@/app/store';
import type { CfaType } from '@/lib/types';
import { RAW_EXTENSIONS } from '@/lib/catalog';
import { extractRafThumbnail, extractRafQuickMetadata } from '@/pipeline/decode/raf-thumbnail';
import { writeRaw, writeThumbnail, deleteAllForFile, listRawFileIds } from '@/app/storage/opfs-storage';
import { deleteFile as idbDeleteFile, cancelPendingPut } from '@/app/storage/idb-storage';
import { matchLens } from '@/app/lens/lensfun';
import { discardResult } from '@/app/services/processing';

function isRawFile(file: File): boolean {
  const lower = file.name.toLowerCase();
  return RAW_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

function newEntry(file: File): QueuedFile {
  return {
    id: crypto.randomUUID(),
    file,
    name: file.name.replace(/\.[^.]+$/, ''),
    originalName: file.name,
    thumbnailUrl: null,
    metadata: null,
    cfaType: (file.name.toLowerCase().endsWith('.raf') ? 'xtrans' : 'bayer') as CfaType,
    status: 'queued',
    error: null,
    progress: null,
    result: null,
    resultMethod: null,
    lensProfile: null,
    lookPreset: 'default',
    openDrtOverrides: {},
    preProcessOverrides: {},
  };
}

/** Add dropped/picked files: entries into the store, bytes and thumbnails into OPFS, lenses matched. */
export function importFiles(files: File[]): void {
  const entries = files.filter(isRawFile).map(newEntry);
  if (entries.length === 0) return;
  useAppStore.getState().addFiles(entries);

  for (const entry of entries) {
    entry.file!.arrayBuffer().then(async (buf) => {
      const exists = () => useAppStore.getState().files.some((file) => file.id === entry.id);
      // Removal can happen before disk reads or writes finish.
      if (!exists()) return;
      const thumbBlob = extractRafThumbnail(buf);
      const meta = extractRafQuickMetadata(buf);
      const writes = [writeRaw(entry.id, buf).catch((e) => console.warn('OPFS raw write failed:', e))];
      if (thumbBlob) {
        writes.push(writeThumbnail(entry.id, thumbBlob).catch((e) => console.warn('OPFS thumbnail write failed:', e)));
      }
      useAppStore.getState().setFileThumbnail(entry.id, thumbBlob ? URL.createObjectURL(thumbBlob) : null, meta);
      matchLensFor(entry.id);
      await Promise.all(writes);
      // A write already in progress must not resurrect storage after removal.
      if (!exists()) await deleteAllForFile(entry.id).catch((e) => console.warn('OPFS cleanup failed:', e));
    }).catch((e) => console.warn('RAW import failed:', e));
  }
}

/** Match the file's lens against the LensFun database if it has lens metadata and no profile yet. */
export function matchLensFor(fileId: string): void {
  const file = useAppStore.getState().files.find((f) => f.id === fileId);
  if (!file || file.lensProfile || !file.metadata?.lensModel) return;
  matchLens(file.metadata.camera, file.metadata.lensModel)
    .then((profile) => {
      if (profile) useAppStore.getState().setFileLensProfile(fileId, profile);
    })
    .catch((e) => console.warn('Lens match failed:', e));
}

export function removeFile(id: string): void {
  const thumbnailUrl = useAppStore.getState().files.find((file) => file.id === id)?.thumbnailUrl;
  if (thumbnailUrl) URL.revokeObjectURL(thumbnailUrl);
  discardResult(id);
  cancelPendingPut(id);
  deleteAllForFile(id).catch((e) => console.warn('OPFS cleanup failed:', e));
  idbDeleteFile(id).catch((e) => console.warn('IDB cleanup failed:', e));
  useAppStore.getState().removeFile(id);
}

/**
 * Remove OPFS entries that no library entry owns (fire-and-forget on startup). Ownership is read
 * from the live store when each entry is considered, so files imported while startup was still
 * running are kept. Call only after the stored library has been restored into the store.
 */
export async function cleanupOrphans(): Promise<void> {
  try {
    const rawIds = await listRawFileIds();
    const owned = new Set(useAppStore.getState().files.map((f) => f.id));
    for (const id of rawIds) {
      if (!owned.has(id)) {
        deleteAllForFile(id).catch(() => {});
      }
    }
  } catch {
    // Non-critical — silently ignore
  }
}
