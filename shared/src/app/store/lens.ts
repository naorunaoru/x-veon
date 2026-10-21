import type { LensProfile } from '@/app/lens/lensfun';
import { effectiveLensEdit, type LensEdit } from '@/lib/lens/edit';
import type { Slice, QueuedFile, AppState } from './types';
import { editPhoto } from './photo';

export interface LensSlice {
  /** Lensfun lenses the user picked by hand, keyed by lensPickKey(file, model); session-only. */
  lensPicks: Record<string, LensProfile>;
  setLensPick: (key: string, profile: LensProfile) => void;
  setFileLens: (fileId: string, patch: Partial<LensEdit>) => void;
  resetFileLens: (fileId: string) => void;
  markLensMatched: (fileId: string) => void;
}

export const lensPickKey = (file: string, model: string): string => `${file}#${model}`;

export const createLensSlice: Slice<LensSlice> = (set) => {
  const update = (id: string, fn: (file: QueuedFile, state: AppState) => QueuedFile) =>
    set((state) => ({
      files: state.files.map((f) => (f.id === id && f.editing !== 'view-only' ? fn(f, state) : f)),
    }));
  return {
    lensPicks: {},
    setLensPick: (key, profile) => set((state) => ({ lensPicks: { ...state.lensPicks, [key]: profile } })),
    setFileLens: (id, patch) =>
      update(id, (f, state) => editPhoto(f, { lens: { ...effectiveLensEdit(f.edit), ...patch } }, state)),
    resetFileLens: (id) => update(id, (f, state) => editPhoto(f, { lens: undefined }, state)),
    markLensMatched: (id) =>
      set((state) => ({ files: state.files.map((f) => (f.id === id ? { ...f, lensMatched: true } : f)) })),
  };
};
