import type { PhotoEdit, LibraryPhoto } from '@/host';
import type { ModelIdentity } from '@/lib/types';
import type { StateCreator } from 'zustand';
import type { CfaType, DemosaicMethod, LookPreset, ProcessingResultMeta } from '@/lib/types';
import type { OpenDrtConfig, PreProcessConfig } from '@/renderer/grading/opendrt-params';
import type { QuickMetadata } from '@/pipeline/decode/raf-thumbnail';
import type { LensProfile } from '@/app/lens/lensfun';
import type { InitSlice } from './init';
import type { LibrarySlice } from './library';
import type { SettingsSlice } from './settings';
import type { GradingSlice } from './grading';
import type { DisplaySlice } from './display';
import type { ViewSlice } from './view';

export type FileStatus = 'queued' | 'processing' | 'done' | 'error';

export interface LookSnapshot {
  lookPreset: LookPreset;
  openDrtOverrides: Partial<OpenDrtConfig>;
}

export interface QueuedFile {
  id: string;
  fileSize: number;
  edit: PhotoEdit;
  editing: LibraryPhoto['editing'];
  editingNote: string | null;
  editRevision: number;
  modelNeedsResolution: boolean;
  actualModel: ModelIdentity | null;
  modelNote: string | null;
  processedKey: string | null;
  name: string;
  originalName: string;
  thumbnailUrl: string | null;
  metadata: QuickMetadata | null;
  cfaType: CfaType | null;
  status: FileStatus;
  error: string | null;
  progress: { current: number; total: number } | null;
  result: ProcessingResultMeta | null;
  resultMethod: DemosaicMethod | null;
  lensProfile: LensProfile | null;
  /** Session-only history of look selections; never serialized with the photo. */
  lookHistory?: LookSnapshot[];
}

export interface ViewControls {
  zoomTo: (scale: number) => void;
  resetView: () => void;
  panTo: (pan: { x: number; y: number }) => void;
}

export type AppState = InitSlice & LibrarySlice & SettingsSlice & GradingSlice & DisplaySlice & ViewSlice;

/** A slice creator that can read and set the whole state (selectFile sets settings, for example). */
export type Slice<T> = StateCreator<AppState, [], [], T>;
