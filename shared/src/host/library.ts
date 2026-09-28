import type { CfaType, DemosaicMethod, LookPreset, ModelIdentity, SerializableResultMeta } from '@/lib/types';
import type { OpenDrtConfig, PreProcessConfig } from '@/renderer/grading/opendrt-params';
import type { QuickMetadata } from '@/pipeline/decode/raf-thumbnail';
import type { LensProfile } from '@/app/lens/lensfun';
export type PhotoId = string;
export interface PhotoEdit {
  version: 1;
  lookPreset: LookPreset;
  openDrtOverrides: Partial<OpenDrtConfig>;
  preProcessOverrides: Partial<PreProcessConfig>;
  demosaicMethod: DemosaicMethod | null;
  model: ModelIdentity | null;
}
export interface PhotoFacts {
  cfaType: CfaType | null;
  metadata: QuickMetadata | null;
  resultMeta: SerializableResultMeta | null;
  lensProfile: LensProfile | null;
  status: 'queued' | 'done' | 'error';
  error: string | null;
}
export interface LibraryPhoto {
  id: PhotoId;
  name: string;
  originalName: string;
  fileSize: number;
  thumbnailUrl: string | null;
  edit: PhotoEdit;
  facts: PhotoFacts;
  editing: 'saved' | 'session' | 'view-only';
  editingNote: string | null;
}
export interface LibrarySnapshot {
  photos: LibraryPhoto[];
  complete: boolean;
  selectedIds?: PhotoId[];
}
export interface FolderRef {
  id: string;
  name: string;
}
export interface LibraryChange {
  snapshot: LibrarySnapshot;
  kind?: 'facts' | 'replace';
}
export interface LibraryHost {
  load(): Promise<LibrarySnapshot>;
  readRaw(id: PhotoId): Promise<ArrayBuffer>;
  save(id: PhotoId, edit: PhotoEdit, facts: PhotoFacts): Promise<void>;
  addFiles(files: File[]): Promise<LibrarySnapshot>;
  onChange?(listener: (change: LibraryChange) => void): () => void;
  openFolder?(): Promise<LibrarySnapshot | null>;
  recentFolders?(): Promise<FolderRef[]>;
  remove?(id: PhotoId): Promise<void>;
  clear?(): Promise<void>;
  /** Release host-owned thumbnail URLs when a snapshot cannot be consumed. */
  release?(photos: { thumbnailUrl: string | null }[]): void;
}
