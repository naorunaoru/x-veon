/**
 * Writes what changes in the store to IndexedDB (file records debounced, settings at once) and
 * restores the library on startup. The only module that writes records or settings.
 */
import { useAppStore } from '@/app/store';
import type { QueuedFile, RestoredSettings } from '@/app/store';
import { serializeResultMeta, deserializeResultMeta } from '@/lib/types';
import type { DemosaicMethod, ExportFormat } from '@/lib/types';
import type { OpenDrtConfig, PreProcessConfig } from '@/renderer/grading/opendrt-params';
import { getAllFiles, getSetting, putSetting, debouncedPutFile } from '@/app/storage/idb-storage';
import type { PersistedFile } from '@/app/storage/idb-storage';
import { readThumbnail } from '@/app/storage/opfs-storage';

const SETTING_KEYS = ['demosaicMethod', 'modelSize', 'exportFormat', 'exportQuality', 'selectedFileId'] as const;

export function fileToPersistedFile(f: QueuedFile): PersistedFile {
  return {
    id: f.id,
    name: f.name,
    originalName: f.originalName,
    fileSize: f.file?.size ?? 0,
    cfaType: f.cfaType,
    camera: f.metadata?.camera ?? null,
    lensModel: f.metadata?.lensModel ?? null,
    focalLength: f.metadata?.focalLength ?? null,
    fNumber: f.metadata?.fNumber ?? null,
    status: f.status === 'processing' ? 'queued' : f.status,
    error: f.error,
    resultMethod: f.resultMethod,
    resultMeta: f.result ? serializeResultMeta(f.result) : null,
    cachedMethods: [],
    lensProfile: f.lensProfile,
    lookPreset: f.lookPreset,
    openDrtOverrides: f.openDrtOverrides as Record<string, number | boolean>,
    preProcessOverrides: f.preProcessOverrides as Record<string, number>,
    addedAt: Date.now(),
  };
}

/** Subscribe to the store; returns the unsubscribe function. */
export function startPersistence(): () => void {
  let previous = useAppStore.getState();
  return useAppStore.subscribe((state) => {
    const before = previous;
    previous = state;

    for (const key of SETTING_KEYS) {
      if (state[key] !== before[key]) putSetting(key, state[key]).catch(() => {});
    }

    if (state.files !== before.files) {
      for (const file of state.files) {
        const old = before.files.find((candidate) => candidate.id === file.id);
        // Transient: the record is written when the run ends (done or error)
        if (old === file || file.status === 'processing') continue;
        debouncedPutFile(fileToPersistedFile(file));
      }
    }
  });
}

async function persistedToQueued(p: PersistedFile): Promise<QueuedFile> {
  // Load thumbnail from OPFS
  const thumbBlob = await readThumbnail(p.id).catch(() => null);

  return {
    id: p.id,
    file: null,
    name: p.name,
    originalName: p.originalName,
    thumbnailUrl: thumbBlob ? URL.createObjectURL(thumbBlob) : null,
    metadata: p.camera ? {
      camera: p.camera,
      lensModel: p.lensModel ?? '',
      focalLength: p.focalLength ?? 0,
      fNumber: p.fNumber ?? 0,
    } : null,
    cfaType: p.cfaType,
    status: p.status === 'done' && p.resultMeta ? 'done' : 'queued',
    error: null,
    progress: null,
    result: p.resultMeta ? deserializeResultMeta(p.resultMeta) : null,
    resultMethod: p.resultMethod,
    lensProfile: p.lensProfile ?? null,
    lookPreset: p.lookPreset,
    openDrtOverrides: p.openDrtOverrides as Partial<OpenDrtConfig>,
    preProcessOverrides: (p.preProcessOverrides ?? {}) as Partial<PreProcessConfig>,
  };
}

/** Read the library and settings back; the caller feeds them to restoreFromDb. */
export async function restore(): Promise<{ files: QueuedFile[]; settings: RestoredSettings }> {
  const [persistedFiles, demosaicMethod, exportFormat, exportQuality, selectedFileId] = await Promise.all([
    getAllFiles().catch(() => [] as PersistedFile[]),
    getSetting<DemosaicMethod>('demosaicMethod').catch(() => undefined),
    getSetting<ExportFormat>('exportFormat').catch(() => undefined),
    getSetting<number>('exportQuality').catch(() => undefined),
    getSetting<string | null>('selectedFileId').catch(() => undefined),
  ]);
  const files: QueuedFile[] = [];
  for (const p of persistedFiles) files.push(await persistedToQueued(p));
  return { files, settings: { demosaicMethod, exportFormat, exportQuality, selectedFileId: selectedFileId ?? undefined } };
}
