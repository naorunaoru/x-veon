import type { LookPreset } from '@/lib/types';
import type { OpenDrtConfig, PreProcessConfig } from '@/renderer/grading/opendrt-params';
import type { Slice } from './types';

export interface GradingSlice {
  setFileLookPreset: (fileId: string, preset: LookPreset) => void;
  setFileOpenDrtOverride: <K extends keyof OpenDrtConfig>(fileId: string, key: K, value: OpenDrtConfig[K]) => void;
  resetFileOpenDrtOverrides: (fileId: string) => void;
  setFilePreProcessOverride: <K extends keyof PreProcessConfig>(fileId: string, key: K, value: PreProcessConfig[K]) => void;
  resetFilePreProcessOverrides: (fileId: string) => void;
  clearFileOpenDrtOverrides: (fileId: string, keys: (keyof OpenDrtConfig)[]) => void;
  clearFilePreProcessOverrides: (fileId: string, keys: (keyof PreProcessConfig)[]) => void;
}

export const createGradingSlice: Slice<GradingSlice> = (set) => ({
  setFileLookPreset: (fileId, preset) =>
    set((state) => ({
      files: state.files.map((f) => (
        f.id === fileId ? { ...f, lookPreset: preset, openDrtOverrides: {} as Partial<OpenDrtConfig> } : f  // preProcessOverrides preserved
      )),
    })),
  setFileOpenDrtOverride: (fileId, key, value) =>
    set((state) => ({
      files: state.files.map((f) => (f.id === fileId ? { ...f, openDrtOverrides: { ...f.openDrtOverrides, [key]: value } } : f)),
    })),
  resetFileOpenDrtOverrides: (fileId) =>
    set((state) => ({
      files: state.files.map((f) => (f.id === fileId ? { ...f, openDrtOverrides: {} as Partial<OpenDrtConfig> } : f)),
    })),
  setFilePreProcessOverride: (fileId, key, value) =>
    set((state) => ({
      files: state.files.map((f) => (f.id === fileId ? { ...f, preProcessOverrides: { ...f.preProcessOverrides, [key]: value } } : f)),
    })),
  resetFilePreProcessOverrides: (fileId) =>
    set((state) => ({
      files: state.files.map((f) => (f.id === fileId ? { ...f, preProcessOverrides: {} as Partial<PreProcessConfig> } : f)),
    })),
  clearFileOpenDrtOverrides: (fileId, keys) =>
    set((state) => ({
      files: state.files.map((f) => {
        if (f.id !== fileId) return f;
        const openDrtOverrides = { ...f.openDrtOverrides };
        for (const k of keys) delete openDrtOverrides[k];
        return { ...f, openDrtOverrides };
      }),
    })),
  clearFilePreProcessOverrides: (fileId, keys) =>
    set((state) => ({
      files: state.files.map((f) => {
        if (f.id !== fileId) return f;
        const preProcessOverrides = { ...f.preProcessOverrides };
        for (const k of keys) delete preProcessOverrides[k];
        return { ...f, preProcessOverrides };
      }),
    })),
});
