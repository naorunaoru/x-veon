import { create } from 'zustand';
import type { DemosaicMethod, ExportFormat, ProcessingResult } from './pipeline/types';
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
  status: FileStatus;
  error: string | null;
  progress: { current: number; total: number } | null;
  result: ProcessingResult | null;
}

interface AppState {
  // Initialization
  initialized: boolean;
  initError: string | null;
  backend: string | null;
  hdrSupported: boolean;
  modelMeta: ModelMeta;

  // File management
  files: QueuedFile[];
  selectedFileId: string | null;

  // Processing settings
  demosaicMethod: DemosaicMethod;

  // Export settings
  exportFormat: ExportFormat;
  exportQuality: number;

  // Canvas ref for WebCodecs AVIF export
  canvasRef: HTMLCanvasElement | null;

  // Actions
  setInitialized: (backend: string, hdrSupported: boolean, modelMeta: ModelMeta) => void;
  setInitError: (error: string) => void;
  addFiles: (files: File[]) => void;
  removeFile: (id: string) => void;
  selectFile: (id: string | null) => void;
  updateFileStatus: (id: string, status: FileStatus, error?: string) => void;
  updateFileProgress: (id: string, current: number, total: number) => void;
  setFileResult: (id: string, result: ProcessingResult) => void;
  setDemosaicMethod: (method: DemosaicMethod) => void;
  setExportFormat: (format: ExportFormat) => void;
  setExportQuality: (quality: number) => void;
  setCanvasRef: (ref: HTMLCanvasElement | null) => void;
}

export const useAppStore = create<AppState>((set, get) => ({
  initialized: false,
  initError: null,
  backend: null,
  hdrSupported: false,
  modelMeta: {},

  files: [],
  selectedFileId: null,

  demosaicMethod: 'neural-net',

  exportFormat: 'avif',
  exportQuality: 85,

  canvasRef: null,

  setInitialized: (backend, hdrSupported, modelMeta) =>
    set({ initialized: true, backend, hdrSupported, modelMeta }),

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
        status: 'queued' as const,
        error: null,
        progress: null,
        result: null,
      }));

    if (entries.length === 0) return;

    const currentFiles = get().files;
    const shouldSelect = currentFiles.length === 0;

    set({
      files: [...currentFiles, ...entries],
      ...(shouldSelect ? { selectedFileId: entries[0].id } : {}),
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
              ...(status === 'processing' ? { result: null, progress: null } : {}),
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

  setFileResult: (id, result) =>
    set((state) => ({
      files: state.files.map((f) =>
        f.id === id ? { ...f, result, status: 'done', progress: null } : f,
      ),
    })),

  setDemosaicMethod: (method) => set({ demosaicMethod: method }),
  setExportFormat: (format) => set({ exportFormat: format }),
  setExportQuality: (quality) => set({ exportQuality: quality }),
  setCanvasRef: (ref) => set({ canvasRef: ref }),
}));
