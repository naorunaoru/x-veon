import type { LookPreset } from '@/lib/types';
import type { OpenDrtConfig, PreProcessConfig } from '@/renderer/grading/opendrt-params';
import type { Slice, QueuedFile } from './types';
import { editPhoto } from './photo';
export interface GradingSlice {
  setFileLookPreset: (fileId: string, preset: LookPreset) => void;
  undoFileLook: (fileId: string) => void;
  setFileOpenDrtOverride: <K extends keyof OpenDrtConfig>(
    fileId: string,
    key: K,
    value: OpenDrtConfig[K],
  ) => void;
  resetFileOpenDrtOverrides: (fileId: string) => void;
  setFilePreProcessOverride: <K extends keyof PreProcessConfig>(
    fileId: string,
    key: K,
    value: PreProcessConfig[K],
  ) => void;
  resetFilePreProcessOverrides: (fileId: string) => void;
  clearFileOpenDrtOverrides: (fileId: string, keys: (keyof OpenDrtConfig)[]) => void;
  clearFilePreProcessOverrides: (fileId: string, keys: (keyof PreProcessConfig)[]) => void;
}
export const createGradingSlice: Slice<GradingSlice> = (set) => {
  const update = (id: string, fn: (file: QueuedFile) => QueuedFile) =>
    set((state) => ({
      files: state.files.map((f) => (f.id === id && f.editing !== 'view-only' ? fn(f) : f)),
    }));
  return {
    setFileLookPreset: (id, preset) =>
      update(id, (f) => {
        if (f.edit.lookPreset === preset && Object.keys(f.edit.openDrtOverrides).length === 0) return f;
        return {
          ...editPhoto(f, { lookPreset: preset, openDrtOverrides: {} }),
          lookHistory: [
            ...(f.lookHistory ?? []),
            { lookPreset: f.edit.lookPreset, openDrtOverrides: { ...f.edit.openDrtOverrides } },
          ].slice(-20),
        };
      }),
    undoFileLook: (id) =>
      update(id, (f) => {
        if (!f.lookHistory?.length) return f;
        return {
          ...editPhoto(f, f.lookHistory[f.lookHistory.length - 1]),
          lookHistory: f.lookHistory.slice(0, -1),
        };
      }),
    setFileOpenDrtOverride: (id, key, value) =>
      update(id, (f) => editPhoto(f, { openDrtOverrides: { ...f.edit.openDrtOverrides, [key]: value } })),
    resetFileOpenDrtOverrides: (id) => update(id, (f) => editPhoto(f, { openDrtOverrides: {} })),
    setFilePreProcessOverride: (id, key, value) =>
      update(id, (f) =>
        editPhoto(f, { preProcessOverrides: { ...f.edit.preProcessOverrides, [key]: value } }),
      ),
    resetFilePreProcessOverrides: (id) => update(id, (f) => editPhoto(f, { preProcessOverrides: {} })),
    clearFileOpenDrtOverrides: (id, keys) =>
      update(id, (f) => {
        const overrides = { ...f.edit.openDrtOverrides };
        for (const key of keys) delete overrides[key];
        return editPhoto(f, { openDrtOverrides: overrides });
      }),
    clearFilePreProcessOverrides: (id, keys) =>
      update(id, (f) => {
        const overrides = { ...f.edit.preProcessOverrides };
        for (const key of keys) delete overrides[key];
        return editPhoto(f, { preProcessOverrides: overrides });
      }),
  };
};
