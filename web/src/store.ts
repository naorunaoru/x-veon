import { create } from 'zustand';
import type { CfaType, DemosaicMethod, ExportFormat, LookPreset, ProcessingResultMeta } from './pipeline/types';
import { serializeResultMeta } from './pipeline/types';
import type { OpenDrtConfig } from './gl/opendrt-params';
import { deleteAllForFile, writeRaw, writeThumbnail } from './lib/opfs-storage';
import { putFile, deleteFile as idbDeleteFile, debouncedPutFile, putSetting } from './lib/idb-storage';
import type { PersistedFile } from './lib/idb-storage';
import type { ModelMeta } from './pipeline/inference';
import { extractRafThumbnail, extractRafQuickMetadata } from './pipeline/raf-thumbnail';
import { RAW_EXTENSIONS } from './pipeline/constants';

export type FileStatus = 'queued' | 'processing' | 'done' | 'error';

export interface QueuedFile {
  id: string;
  file: File | null;
  name: string;
  originalName: string;
  thumbnailUrl: string | null;
  metadata: { camera: string } | null;
  cfaType: CfaType | null;
  status: FileStatus;
  error: string | null;
  progress: { current: number; total: number } | null;
  result: ProcessingResultMeta | null;
  resultMethod: DemosaicMethod | null;
  cachedResults: Partial<Record<DemosaicMethod, ProcessingResultMeta>>;
  lookPreset: LookPreset;
  openDrtOverrides: Partial<OpenDrtConfig>;
}

interface AppState {
  // Initialization
  initialized: boolean;
  initError: string | null;
  backend: string | null;
  modelMeta: ModelMeta;

  // File management
  files: QueuedFile[];
  selectedFileId: string | null;

  // Processing settings
  demosaicMethod: DemosaicMethod;

  // Export settings
  exportFormat: ExportFormat;
  exportQuality: number;

  // HDR display output (WebGL2 extended range)
  displayHdr: boolean;
  displayHdrHeadroom: number;

  // Canvas ref for WebCodecs AVIF export
  canvasRef: HTMLCanvasElement | null;

  // Actions
  setInitialized: (backend: string, modelMeta: ModelMeta) => void;
  setInitError: (error: string) => void;
  addFiles: (files: File[]) => void;
  removeFile: (id: string) => void;
  selectFile: (id: string | null) => void;
  updateFileStatus: (id: string, status: FileStatus, error?: string) => void;
  updateFileProgress: (id: string, current: number, total: number) => void;
  setFileResult: (id: string, result: ProcessingResultMeta, method: DemosaicMethod) => void;
  restoreCachedResult: (id: string, method: DemosaicMethod) => void;
  setDemosaicMethod: (method: DemosaicMethod) => void;
  setExportFormat: (format: ExportFormat) => void;
  setExportQuality: (quality: number) => void;

  // Per-file grading
  setFileLookPreset: (fileId: string, preset: LookPreset) => void;
  setFileOpenDrtOverride: <K extends keyof OpenDrtConfig>(fileId: string, key: K, value: OpenDrtConfig[K]) => void;
  resetFileOpenDrtOverrides: (fileId: string) => void;

  setDisplayHdr: (enabled: boolean, headroom: number) => void;
  setCanvasRef: (ref: HTMLCanvasElement | null) => void;

  // Restore from IndexedDB on startup
  restoreFromDb: (files: QueuedFile[], settings: {
    demosaicMethod?: DemosaicMethod;
    exportFormat?: ExportFormat;
    exportQuality?: number;
    selectedFileId?: string | null;
  }) => void;
}

// ── Persistence helpers ─────────────────────────────────────────────────────

function fileToPersistedFile(f: QueuedFile): PersistedFile {
  return {
    id: f.id,
    name: f.name,
    originalName: f.originalName,
    fileSize: f.file?.size ?? 0,
    cfaType: f.cfaType,
    camera: f.metadata?.camera ?? null,
    status: f.status === 'processing' ? 'queued' : f.status,
    error: f.error,
    resultMethod: f.resultMethod,
    resultMeta: f.result ? serializeResultMeta(f.result) : null,
    cachedMethods: Object.keys(f.cachedResults) as DemosaicMethod[],
    lookPreset: f.lookPreset,
    openDrtOverrides: f.openDrtOverrides as Record<string, number | boolean>,
    addedAt: Date.now(),
  };
}

function persistFile(f: QueuedFile): void {
  putFile(fileToPersistedFile(f)).catch((e) => console.warn('IDB persist failed:', e));
}

function persistFileDebounced(f: QueuedFile): void {
  debouncedPutFile(fileToPersistedFile(f));
}

// ── Store ───────────────────────────────────────────────────────────────────

export const useAppStore = create<AppState>((set, get) => ({
  initialized: false,
  initError: null,
  backend: null,
  modelMeta: {},

  files: [],
  selectedFileId: null,

  demosaicMethod: 'neural-net',

  exportFormat: 'jpeg-hdr',
  exportQuality: 95,

  displayHdr: false,
  displayHdrHeadroom: 1.0,

  canvasRef: null,

  setInitialized: (backend, modelMeta) =>
    set({ initialized: true, backend, modelMeta }),

  setInitError: (error) =>
    set({ initError: error }),

  addFiles: (newFiles) => {
    const entries: QueuedFile[] = newFiles
      .filter((f) => {
        const lower = f.name.toLowerCase();
        return RAW_EXTENSIONS.some((ext) => lower.endsWith(ext));
      })
      .map((f) => ({
        id: crypto.randomUUID(),
        file: f,
        name: f.name.replace(/\.[^.]+$/, ''),
        originalName: f.name,
        thumbnailUrl: null,
        metadata: null,
        cfaType: (f.name.toLowerCase().endsWith('.raf') ? 'xtrans' : 'bayer') as CfaType,
        status: 'queued' as const,
        error: null,
        progress: null,
        result: null,
        resultMethod: null,
        cachedResults: {},
        lookPreset: 'default' as const,
        openDrtOverrides: {},
      }));

    if (entries.length === 0) return;

    const currentFiles = get().files;

    set({
      files: [...currentFiles, ...entries],
      selectedFileId: entries[0].id,
    });

    putSetting('selectedFileId', entries[0].id).catch(() => {});

    // Write RAW to OPFS + persist to IDB, extract thumbnails
    for (const entry of entries) {
      entry.file!.arrayBuffer().then((buf) => {
        // Write raw to OPFS
        writeRaw(entry.id, buf).catch((e) => console.warn('OPFS raw write failed:', e));

        // Extract thumbnail and metadata
        const thumbBlob = extractRafThumbnail(buf);
        const meta = extractRafQuickMetadata(buf);

        // Write thumbnail to OPFS
        if (thumbBlob) {
          writeThumbnail(entry.id, thumbBlob).catch((e) => console.warn('OPFS thumbnail write failed:', e));
        }

        set((state) => ({
          files: state.files.map((f) =>
            f.id === entry.id
              ? {
                  ...f,
                  thumbnailUrl: thumbBlob ? URL.createObjectURL(thumbBlob) : null,
                  metadata: meta,
                }
              : f,
          ),
        }));

        // Persist metadata to IDB
        const updated = get().files.find((f) => f.id === entry.id);
        if (updated) {
          putFile(fileToPersistedFile(updated)).catch((e) => console.warn('IDB persist failed:', e));
        }
      });
    }
  },

  removeFile: (id) => {
    deleteAllForFile(id).catch((e) => console.warn('OPFS cleanup failed:', e));
    idbDeleteFile(id).catch((e) => console.warn('IDB cleanup failed:', e));
    set((state) => {
      const files = state.files.filter((f) => f.id !== id);
      const selectedFileId =
        state.selectedFileId === id
          ? files.length > 0
            ? files[0].id
            : null
          : state.selectedFileId;
      putSetting('selectedFileId', selectedFileId).catch(() => {});
      return { files, selectedFileId };
    });
  },

  selectFile: (id) => {
    set({ selectedFileId: id });
    putSetting('selectedFileId', id).catch(() => {});
  },

  updateFileStatus: (id, status, error) =>
    set((state) => ({
      files: state.files.map((f) => {
        if (f.id !== id) return f;
        const updated = {
          ...f,
          status,
          error: error ?? null,
          ...(status === 'processing' ? { progress: null } : {}),
        };
        // Persist non-transient status changes
        if (status === 'error') persistFile(updated);
        return updated;
      }),
    })),

  updateFileProgress: (id, current, total) =>
    set((state) => ({
      files: state.files.map((f) =>
        f.id === id ? { ...f, progress: { current, total } } : f,
      ),
    })),

  setFileResult: (id, result, method) =>
    set((state) => ({
      files: state.files.map((f) => {
        if (f.id !== id) return f;
        const updated = {
          ...f,
          result,
          resultMethod: method,
          cachedResults: method === 'neural-net'
            ? { ...f.cachedResults, [method]: result }
            : f.cachedResults,
          status: 'done' as const,
          progress: null,
        };
        persistFile(updated);
        return updated;
      }),
    })),

  restoreCachedResult: (id, method) =>
    set((state) => ({
      files: state.files.map((f) => {
        if (f.id !== id) return f;
        const cached = f.cachedResults[method];
        if (!cached) return f;
        return { ...f, result: cached, resultMethod: method, status: 'done' as const, progress: null };
      }),
    })),

  setDemosaicMethod: (method) => {
    set({ demosaicMethod: method });
    putSetting('demosaicMethod', method).catch(() => {});
  },
  setExportFormat: (format) => {
    set({ exportFormat: format });
    putSetting('exportFormat', format).catch(() => {});
  },
  setExportQuality: (quality) => {
    set({ exportQuality: quality });
    putSetting('exportQuality', quality).catch(() => {});
  },

  // Per-file grading
  setFileLookPreset: (fileId, preset) =>
    set((state) => ({
      files: state.files.map((f) => {
        if (f.id !== fileId) return f;
        const updated = { ...f, lookPreset: preset, openDrtOverrides: {} as Partial<OpenDrtConfig> };
        persistFileDebounced(updated);
        return updated;
      }),
    })),

  setFileOpenDrtOverride: (fileId, key, value) =>
    set((state) => ({
      files: state.files.map((f) => {
        if (f.id !== fileId) return f;
        const updated = { ...f, openDrtOverrides: { ...f.openDrtOverrides, [key]: value } };
        persistFileDebounced(updated);
        return updated;
      }),
    })),

  resetFileOpenDrtOverrides: (fileId) =>
    set((state) => ({
      files: state.files.map((f) => {
        if (f.id !== fileId) return f;
        const updated = { ...f, openDrtOverrides: {} as Partial<OpenDrtConfig> };
        persistFile(updated);
        return updated;
      }),
    })),

  setDisplayHdr: (enabled, headroom) => set({ displayHdr: enabled, displayHdrHeadroom: headroom }),
  setCanvasRef: (ref) => set({ canvasRef: ref }),

  restoreFromDb: (files, settings) =>
    set({
      files,
      selectedFileId: settings.selectedFileId ?? (files.length > 0 ? files[0].id : null),
      demosaicMethod: settings.demosaicMethod ?? 'neural-net',
      exportFormat: settings.exportFormat ?? 'jpeg-hdr',
      exportQuality: settings.exportQuality ?? 95,
    }),
}));
