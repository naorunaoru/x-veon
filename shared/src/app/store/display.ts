import type { Slice } from './types';

export interface DisplaySlice {
  displayHdr: boolean;
  displayHdrHeadroom: number;
  hdrPermissionNeeded: boolean;
  setDisplayHdr: (enabled: boolean, headroom: number) => void;
  setHdrPermissionNeeded: (needed: boolean) => void;
}

export const createDisplaySlice: Slice<DisplaySlice> = (set) => ({
  displayHdr: false,
  displayHdrHeadroom: 1.0,
  hdrPermissionNeeded: false,
  setDisplayHdr: (displayHdr, displayHdrHeadroom) => set({ displayHdr, displayHdrHeadroom }),
  setHdrPermissionNeeded: (hdrPermissionNeeded) => set({ hdrPermissionNeeded }),
});
