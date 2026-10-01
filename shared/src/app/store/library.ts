import type { ModelIdentity, ModelSize } from '@/lib/types';
import { editPhoto } from './photo';
import type { DemosaicMethod, ExportFormat, ProcessingResultMeta } from '@/lib/types';
import type { QuickMetadata } from '@/pipeline/decode/raf-thumbnail';
import type { LensProfile } from '@/app/lens/lensfun';
import type { AppState, FileStatus, QueuedFile, Slice } from './types';

export interface RestoredSettings {
  demosaicMethod?: DemosaicMethod;
  modelSize?: ModelSize;
  exportFormat?: ExportFormat;
  exportQuality?: number;
  selectedFileId?: string | null;
}

export interface LibrarySlice {
  hydrationVersion: number;
  files: QueuedFile[];
  setFileDemosaicMethod: (id: string, method: DemosaicMethod) => void;
  setFileModel: (id: string, model: ModelIdentity) => void;
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
  hydrationVersion: 0,
  files: [],
  setFileDemosaicMethod: (id, method) => set(state => ({ files: state.files.map(f => f.id === id ? editPhoto(f, { demosaicMethod: method }, state) : f) })),
  setFileModel: (id, model) => set(state => ({ files: state.files.map(f => f.id === id ? editPhoto(f, { model, demosaicMethod: 'neural-net' }, state) : f) })),
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

  selectFile: (id) => set({ selectedFileId: id }),

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

  restoreFromDb: (restored, settings) => {
    // Files imported while startup was running are already in the store: keep them after the
    // restored library, and keep their selection.
    const state = get();
    const { files: live, selectedFileId: liveSelection } = state;
    const restoredIds = new Set(restored.map((f) => f.id));
    const files = [...restored, ...live.filter((f) => !restoredIds.has(f.id))];
    const has = (id: string | null | undefined): id is string => !!id && files.some((f) => f.id === id);
    const selectedFileId = has(liveSelection) ? liveSelection
      : has(settings.selectedFileId) ? settings.selectedFileId
      : files.length > 0 ? files[0].id : null;
    set({
      files,
      selectedFileId,
      hydrationVersion: state.hydrationVersion + 1,
      demosaicMethod: settings.demosaicMethod ?? state.demosaicMethod,
      modelSize: settings.modelSize ?? state.modelSize,
      exportFormat: settings.exportFormat ?? state.exportFormat,
      exportQuality: settings.exportQuality ?? state.exportQuality,
    });
  },
});
