import { create } from 'zustand';
import type { CfaType, DemosaicMethod, ExportFormat, LookPreset, ProcessingResultMeta } from './pipeline/types';
import type { OpenDrtConfig } from './gl/opendrt-params';
import { deleteHwcForFile } from './lib/opfs-storage';
import type { ModelMeta } from './pipeline/inference';
import { extractRafThumbnail, extractRafQuickMetadata } from './pipeline/raf-thumbnail';
import { RAW_EXTENSIONS } from './pipeline/constants';

export type FileStatus = 'queued' | 'processing' | 'done' | 'error';

export interface QueuedFile {
  id: string;
  file: File;
  name: string;
  thumbnailUrl: string | null;
  metadata: { camera: string } | null;
  cfaType: CfaType | null;
  status: FileStatus;
  error: string | null;
  progress: { current: number; total: number } | null;
  result: ProcessingResultMeta | null;
  resultMethod: DemosaicMethod | null;
  cachedResults: Partial<Record<DemosaicMethod, ProcessingResultMeta>>;
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
  lookPreset: LookPreset;

  // OpenDRT per-parameter overrides (merged on top of preset)
  openDrtOverrides: Partial<OpenDrtConfig>;

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
  setLookPreset: (preset: LookPreset) => void;
  setOpenDrtOverride: <K extends keyof OpenDrtConfig>(key: K, value: OpenDrtConfig[K]) => void;
  resetOpenDrtOverrides: () => void;
  setDisplayHdr: (enabled: boolean, headroom: number) => void;
  setCanvasRef: (ref: HTMLCanvasElement | null) => void;
}

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
  lookPreset: 'default',
  openDrtOverrides: {},

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
        thumbnailUrl: null,
        metadata: null,
        cfaType: (f.name.toLowerCase().endsWith('.raf') ? 'xtrans' : 'bayer') as CfaType,
        status: 'queued' as const,
        error: null,
        progress: null,
        result: null,
        resultMethod: null,
        cachedResults: {},
      }));

    if (entries.length === 0) return;

    const currentFiles = get().files;

    set({
      files: [...currentFiles, ...entries],
      selectedFileId: entries[0].id,
    });

    // Extract thumbnails and metadata async
    for (const entry of entries) {
      entry.file.arrayBuffer().then((buf) => {
        const thumbBlob = extractRafThumbnail(buf);
        const meta = extractRafQuickMetadata(buf);

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
      });
    }
  },

  removeFile: (id) => {
    deleteHwcForFile(id).catch((e) => console.warn('OPFS cleanup failed:', e));
    set((state) => {
      const files = state.files.filter((f) => f.id !== id);
      const selectedFileId =
        state.selectedFileId === id
          ? files.length > 0
            ? files[0].id
            : null
          : state.selectedFileId;
      return { files, selectedFileId };
    });
  },

  selectFile: (id) => set({ selectedFileId: id }),

  updateFileStatus: (id, status, error) =>
    set((state) => ({
      files: state.files.map((f) =>
        f.id === id
          ? {
              ...f,
              status,
              error: error ?? null,
              ...(status === 'processing' ? { progress: null } : {}),
            }
          : f,
      ),
    })),

  updateFileProgress: (id, current, total) =>
    set((state) => ({
      files: state.files.map((f) =>
        f.id === id ? { ...f, progress: { current, total } } : f,
      ),
    })),

  setFileResult: (id, result, method) =>
    set((state) => ({
      files: state.files.map((f) =>
        f.id === id ? {
          ...f,
          result,
          resultMethod: method,
          // Only cache NN results — traditional methods are fast to recompute
          cachedResults: method === 'neural-net'
            ? { ...f.cachedResults, [method]: result }
            : f.cachedResults,
          status: 'done' as const,
          progress: null,
        } : f,
      ),
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

  setDemosaicMethod: (method) => set({ demosaicMethod: method }),
  setExportFormat: (format) => set({ exportFormat: format }),
  setExportQuality: (quality) => set({ exportQuality: quality }),
  setLookPreset: (preset) => set({ lookPreset: preset, openDrtOverrides: {} }),
  setOpenDrtOverride: (key, value) =>
    set((state) => ({ openDrtOverrides: { ...state.openDrtOverrides, [key]: value } })),
  resetOpenDrtOverrides: () => set({ openDrtOverrides: {} }),
  setDisplayHdr: (enabled, headroom) => set({ displayHdr: enabled, displayHdrHeadroom: headroom }),
  setCanvasRef: (ref) => set({ canvasRef: ref }),
}));
