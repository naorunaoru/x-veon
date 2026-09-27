import { create } from 'zustand';
import type { AppState } from './types';
import { createInitSlice } from './init';
import { createLibrarySlice } from './library';
import { createSettingsSlice } from './settings';
import { createGradingSlice } from './grading';
import { createDisplaySlice } from './display';
import { createViewSlice } from './view';

export type { AppState, FileStatus, QueuedFile, ViewControls } from './types';
export type { RestoredSettings } from './library';

/** Pure state and synchronous setters only. Side effects live in app/services. */
export const useAppStore = create<AppState>()((...a) => ({
  ...createInitSlice(...a),
  ...createLibrarySlice(...a),
  ...createSettingsSlice(...a),
  ...createGradingSlice(...a),
  ...createDisplaySlice(...a),
  ...createViewSlice(...a),
}));
