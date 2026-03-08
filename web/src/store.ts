import { create } from 'zustand';
import type { CfaType, DemosaicMethod, ExportFormat, LookPreset, ModelSize, ProcessingResultMeta } from './pipeline/types';
import { serializeResultMeta } from './pipeline/types';
import type { OpenDrtConfig, PreProcessConfig } from './gl/opendrt-params';
import { deleteAllForFile, writeRaw, writeThumbnail } from './lib/opfs-storage';
import { putFile, deleteFile as idbDeleteFile, debouncedPutFile, putSetting } from './lib/idb-storage';
import type { PersistedFile } from './lib/idb-storage';
import type { HdrRenderer } from './gl/renderer';
import { extractRafThumbnail, extractRafQuickMetadata } from './pipeline/raf-thumbnail';
import type { QuickMetadata } from './pipeline/raf-thumbnail';
import { RAW_EXTENSIONS } from './pipeline/constants';
import { matchLens } from './lib/lensfun';
import type { LensProfile } from './lib/lensfun';

export type FileStatus = 'queued' | 'processing' | 'done' | 'error';

export interface QueuedFile {
  id: string;
  file: File | null;
  name: string;
  originalName: string;
  thumbnailUrl: string | null;
  metadata: QuickMetadata | null;
  cfaType: CfaType | null;
  status: FileStatus;
  error: string | null;
  progress: { current: number; total: number } | null;
  result: ProcessingResultMeta | null;
  resultMethod: DemosaicMethod | null;
  cachedResults: Partial<Record<DemosaicMethod, ProcessingResultMeta>>;
  lensProfile: LensProfile | null;
  lookPreset: LookPreset;
  openDrtOverrides: Partial<OpenDrtConfig>;
  preProcessOverrides: Partial<PreProcessConfig>;
}

interface AppState {
  // Initialization
  initialized: boolean;
  initError: string | null;
  backend: string | null;

  // File management
  files: QueuedFile[];
  selectedFileId: string | null;

  // Processing settings
  modelSize: ModelSize;
  demosaicMethod: DemosaicMethod;

  // Export settings
  exportFormat: ExportFormat;
  exportQuality: number;

  // HDR display output (WebGL2 extended range)
  displayHdr: boolean;
  displayHdrHeadroom: number;
  hdrPermissionNeeded: boolean;

  // Clip mask overlay
  showClipMask: boolean;

  // ML highlight reconstruction: feed clip mask (channel 5) to model's HL head.
  // When false, numeric reconstruction runs instead and channel 5 is zeroed.
  mlHighlightReconstruction: boolean;

  // Canvas ref for WebCodecs AVIF export
  canvasRef: HTMLCanvasElement | null;

  // Renderer ref for GPU export readback
  rendererRef: HdrRenderer | null;

  // Actions
  setInitialized: (backend: string) => void;
  setInitError: (error: string) => void;
  addFiles: (files: File[]) => void;
  removeFile: (id: string) => void;
  selectFile: (id: string | null) => void;
  updateFileStatus: (id: string, status: FileStatus, error?: string) => void;
  updateFileProgress: (id: string, current: number, total: number) => void;
  setFileResult: (id: string, result: ProcessingResultMeta, method: DemosaicMethod) => void;
  restoreCachedResult: (id: string, method: DemosaicMethod) => void;
  setModelSize: (size: ModelSize) => void;
  setDemosaicMethod: (method: DemosaicMethod) => void;
  setExportFormat: (format: ExportFormat) => void;
  setExportQuality: (quality: number) => void;

  // Per-file lens profile
  setFileLensProfile: (fileId: string, profile: LensProfile | null) => void;

  // Per-file grading
  setFileLookPreset: (fileId: string, preset: LookPreset) => void;
  setFileOpenDrtOverride: <K extends keyof OpenDrtConfig>(fileId: string, key: K, value: OpenDrtConfig[K]) => void;
  resetFileOpenDrtOverrides: (fileId: string) => void;
  setFilePreProcessOverride: <K extends keyof PreProcessConfig>(fileId: string, key: K, value: PreProcessConfig[K]) => void;
  resetFilePreProcessOverrides: (fileId: string) => void;

  setDisplayHdr: (enabled: boolean, headroom: number) => void;
  setHdrPermissionNeeded: (needed: boolean) => void;
  setShowClipMask: (show: boolean) => void;
  setMlHighlightReconstruction: (use: boolean) => void;
  setCanvasRef: (ref: HTMLCanvasElement | null) => void;
  setRendererRef: (ref: HdrRenderer | null) => void;

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
    lensModel: f.metadata?.lensModel ?? null,
    focalLength: f.metadata?.focalLength ?? null,
    fNumber: f.metadata?.fNumber ?? null,
    status: f.status === 'processing' ? 'queued' : f.status,
    error: f.error,
    resultMethod: f.resultMethod,
    resultMeta: f.result ? serializeResultMeta(f.result) : null,
    cachedMethods: Object.keys(f.cachedResults) as DemosaicMethod[],
    lensProfile: f.lensProfile,
    lookPreset: f.lookPreset,
    openDrtOverrides: f.openDrtOverrides as Record<string, number | boolean>,
    preProcessOverrides: f.preProcessOverrides as Record<string, number>,
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

  files: [],
  selectedFileId: null,

  modelSize: 'S' as ModelSize,
  demosaicMethod: 'neural-net',

  exportFormat: 'jpeg-hdr',
  exportQuality: 95,

  displayHdr: false,
  displayHdrHeadroom: 1.0,
  hdrPermissionNeeded: false,

  showClipMask: false,
  mlHighlightReconstruction: true,

  canvasRef: null,
  rendererRef: null,

  setInitialized: (backend) =>
    set({ initialized: true, backend }),

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
        lensProfile: null,
        lookPreset: 'default' as const,
        openDrtOverrides: {},
        preProcessOverrides: {},
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

        // Match lens against LensFun database
        if (meta?.lensModel) {
          matchLens(meta.camera, meta.lensModel)
            .then((profile) => {
              if (profile) get().setFileLensProfile(entry.id, profile);
            })
            .catch((e) => console.warn('Lens match failed:', e));
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
    const file = id ? get().files.find((f) => f.id === id) : null;
    const updates: Partial<AppState> = { selectedFileId: id };
    if (file?.resultMethod) {
      updates.demosaicMethod = file.resultMethod;
      putSetting('demosaicMethod', file.resultMethod).catch(() => {});
    }
    // Restore model size and ML HL toggle from the result that produced this file
    const meta = file?.result?.metadata;
    if (meta?.modelSize) {
      updates.modelSize = meta.modelSize;
      putSetting('modelSize', meta.modelSize).catch(() => {});
    }
    if (meta?.mlHighlightReconstruction !== undefined) {
      updates.mlHighlightReconstruction = meta.mlHighlightReconstruction;
    }
    set(updates);
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
        // Merge lens/camera info from processing result into quick metadata
        const rm = result.metadata;
        const camera = f.metadata?.camera || [rm.make, rm.model].filter(Boolean).join(' ');
        const metadata: QuickMetadata = {
          camera,
          lensModel: f.metadata?.lensModel || rm.lensModel,
          focalLength: f.metadata?.focalLength || rm.focalLength,
          fNumber: f.metadata?.fNumber || rm.fNumber,
        };
        const updated = {
          ...f,
          metadata,
          result,
          resultMethod: method,
          cachedResults: method === 'neural-net'
            ? { ...f.cachedResults, [method]: result }
            : f.cachedResults,
          status: 'done' as const,
          progress: null,
        };
        persistFile(updated);

        // Match lens if not already matched and lens info is now available
        if (!f.lensProfile && metadata.lensModel) {
          queueMicrotask(() => {
            matchLens(metadata.camera, metadata.lensModel)
              .then((profile) => {
                if (profile) get().setFileLensProfile(id, profile);
              })
              .catch((e) => console.warn('Lens match failed:', e));
          });
        }

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

  setModelSize: (size) => {
    set({ modelSize: size });
    putSetting('modelSize', size).catch(() => {});
  },
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

  // Per-file lens profile
  setFileLensProfile: (fileId, profile) =>
    set((state) => ({
      files: state.files.map((f) => {
        if (f.id !== fileId) return f;
        const updated = { ...f, lensProfile: profile };
        persistFile(updated);
        return updated;
      }),
    })),

  // Per-file grading
  setFileLookPreset: (fileId, preset) =>
    set((state) => ({
      files: state.files.map((f) => {
        if (f.id !== fileId) return f;
        const updated = { ...f, lookPreset: preset, openDrtOverrides: {} as Partial<OpenDrtConfig> }; // preProcessOverrides preserved
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

  setFilePreProcessOverride: (fileId, key, value) =>
    set((state) => ({
      files: state.files.map((f) => {
        if (f.id !== fileId) return f;
        const updated = { ...f, preProcessOverrides: { ...f.preProcessOverrides, [key]: value } };
        persistFileDebounced(updated);
        return updated;
      }),
    })),

  resetFilePreProcessOverrides: (fileId) =>
    set((state) => ({
      files: state.files.map((f) => {
        if (f.id !== fileId) return f;
        const updated = { ...f, preProcessOverrides: {} as Partial<PreProcessConfig> };
        persistFile(updated);
        return updated;
      }),
    })),

  setDisplayHdr: (enabled, headroom) => set({ displayHdr: enabled, displayHdrHeadroom: headroom }),
  setHdrPermissionNeeded: (needed) => set({ hdrPermissionNeeded: needed }),
  setShowClipMask: (show) => set({ showClipMask: show }),
  setMlHighlightReconstruction: (use) => set({ mlHighlightReconstruction: use }),
  setCanvasRef: (ref) => set({ canvasRef: ref }),
  setRendererRef: (ref) => set({ rendererRef: ref }),

  restoreFromDb: (files, settings) => {
    const selectedFileId = settings.selectedFileId ?? (files.length > 0 ? files[0].id : null);
    const selectedFile = selectedFileId ? files.find((f) => f.id === selectedFileId) : null;
    const meta = selectedFile?.result?.metadata;
    set({
      files,
      selectedFileId,
      demosaicMethod: selectedFile?.resultMethod ?? settings.demosaicMethod ?? 'neural-net',
      exportFormat: settings.exportFormat ?? 'jpeg-hdr',
      exportQuality: settings.exportQuality ?? 95,
      ...(meta?.modelSize ? { modelSize: meta.modelSize } : {}),
      ...(meta?.mlHighlightReconstruction !== undefined ? { mlHighlightReconstruction: meta.mlHighlightReconstruction } : {}),
    });
  },
}));
