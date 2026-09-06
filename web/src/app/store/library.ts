import type { DemosaicMethod, ExportFormat, ProcessingResultMeta } from '@/lib/types';
import type { QuickMetadata } from '@/pipeline/decode/raf-thumbnail';
import type { LensProfile } from '@/app/lens/lensfun';
import type { AppState, FileStatus, QueuedFile, Slice } from './types';

export interface RestoredSettings {
  demosaicMethod?: DemosaicMethod;
  exportFormat?: ExportFormat;
  exportQuality?: number;
  selectedFileId?: string | null;
}

export interface LibrarySlice {
  files: QueuedFile[];
  selectedFileId: string | null;
  processingFileId: string | null;
  /** Append prepared entries and select the first of them. */
  addFiles: (entries: QueuedFile[]) => void;
  removeFile: (id: string) => void;
  selectFile: (id: string | null) => void;
  updateFileStatus: (id: string, status: FileStatus, error?: string) => void;
  updateFileProgress: (id: string, current: number, total: number) => void;
  setFileResult: (id: string, result: ProcessingResultMeta, method: DemosaicMethod) => void;
  setFileThumbnail: (id: string, thumbnailUrl: string | null, metadata: QuickMetadata | null) => void;
  setFileLensProfile: (fileId: string, profile: LensProfile | null) => void;
  setProcessingFileId: (id: string | null) => void;
  restoreFromDb: (files: QueuedFile[], settings: RestoredSettings) => void;
}

export const createLibrarySlice: Slice<LibrarySlice> = (set, get) => ({
  files: [],
  selectedFileId: null,
  processingFileId: null,

  addFiles: (entries) => {
    if (entries.length === 0) return;
    set({ files: [...get().files, ...entries], selectedFileId: entries[0].id });
  },

  removeFile: (id) =>
    set((state) => {
      const removedIndex = state.files.findIndex((f) => f.id === id);
      const files = state.files.filter((f) => f.id !== id);
      const selectedFileId =
        state.selectedFileId === id
          ? files.length > 0
            ? files[Math.min(removedIndex, files.length - 1)].id
            : null
          : state.selectedFileId;
      return { files, selectedFileId };
    }),

  selectFile: (id) => {
    const file = id ? get().files.find((f) => f.id === id) : null;
    const updates: Partial<AppState> = { selectedFileId: id };
    if (file?.resultMethod) updates.demosaicMethod = file.resultMethod;
    // Restore model size from the result that produced this file
    const meta = file?.result?.metadata;
    if (meta?.modelSize) updates.modelSize = meta.modelSize;
    set(updates);
  },

  updateFileStatus: (id, status, error) =>
    set((state) => ({
      files: state.files.map((f) => (
        f.id !== id ? f : { ...f, status, error: error ?? null, ...(status === 'processing' ? { progress: null } : {}) }
      )),
    })),

  updateFileProgress: (id, current, total) =>
    set((state) => ({
      files: state.files.map((f) => (f.id === id ? { ...f, progress: { current, total } } : f)),
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
        return { ...f, metadata, result, resultMethod: method, status: 'done' as const, progress: null };
      }),
    })),

  setFileThumbnail: (id, thumbnailUrl, metadata) =>
    set((state) => ({ files: state.files.map((f) => (f.id === id ? { ...f, thumbnailUrl, metadata } : f)) })),

  setFileLensProfile: (fileId, profile) =>
    set((state) => ({ files: state.files.map((f) => (f.id === fileId ? { ...f, lensProfile: profile } : f)) })),

  setProcessingFileId: (processingFileId) => set({ processingFileId }),

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
    });
  },
});
