import type { Slice } from './types';

export interface InitSlice {
  initialized: boolean;
  initError: string | null;
  backend: string | null;
  setInitialized: (backend: string) => void;
  setInitError: (error: string) => void;
}

export const createInitSlice: Slice<InitSlice> = (set) => ({
  initialized: false,
  initError: null,
  backend: null,
  setInitialized: (backend) => set({ initialized: true, backend }),
  setInitError: (error) => set({ initError: error }),
});
