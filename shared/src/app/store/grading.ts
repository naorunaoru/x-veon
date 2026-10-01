import type { LookPreset } from '@/lib/types';
import type { OpenDrtConfig, PreProcessConfig } from '@/renderer/grading/opendrt-params';
import type { Slice, QueuedFile, AppState } from './types';
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
  const update = (id: string, fn: (file: QueuedFile, state: AppState) => QueuedFile) =>
    set((state) => ({
      files: state.files.map((f) => (f.id === id && f.editing !== 'view-only' ? fn(f, state) : f)),
    }));
  return {
    setFileLookPreset: (id, preset) =>
      update(id, (f, state) => {
        if (f.edit.lookPreset === preset && Object.keys(f.edit.openDrtOverrides).length === 0) return f;
        return {
          ...editPhoto(f, { lookPreset: preset, openDrtOverrides: {} }, state),
          lookHistory: [
            ...(f.lookHistory ?? []),
            { lookPreset: f.edit.lookPreset, openDrtOverrides: { ...f.edit.openDrtOverrides } },
          ].slice(-20),
        };
      }),
    undoFileLook: (id) =>
      update(id, (f, state) => {
        if (!f.lookHistory?.length) return f;
        return {
          ...editPhoto(f, f.lookHistory[f.lookHistory.length - 1], state),
          lookHistory: f.lookHistory.slice(0, -1),
        };
      }),
    setFileOpenDrtOverride: (id, key, value) =>
      update(id, (f, state) => editPhoto(f, { openDrtOverrides: { ...f.edit.openDrtOverrides, [key]: value } }, state)),
    resetFileOpenDrtOverrides: (id) => update(id, (f, state) => editPhoto(f, { openDrtOverrides: {} }, state)),
    setFilePreProcessOverride: (id, key, value) =>
      update(id, (f, state) =>
        editPhoto(f, { preProcessOverrides: { ...f.edit.preProcessOverrides, [key]: value } }, state),
      ),
    resetFilePreProcessOverrides: (id) => update(id, (f, state) => editPhoto(f, { preProcessOverrides: {} }, state)),
    clearFileOpenDrtOverrides: (id, keys) =>
      update(id, (f, state) => {
        const overrides = { ...f.edit.openDrtOverrides };
        for (const key of keys) delete overrides[key];
        return editPhoto(f, { openDrtOverrides: overrides }, state);
      }),
    clearFilePreProcessOverrides: (id, keys) =>
      update(id, (f, state) => {
        const overrides = { ...f.edit.preProcessOverrides };
        for (const key of keys) delete overrides[key];
        return editPhoto(f, { preProcessOverrides: overrides }, state);
      }),
  };
};
