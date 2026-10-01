import type { FolderRef, PhotoId, UnsavedSummary } from '@/host';
export type { UnsavedSummary } from '@/host';
export type BridgeEvent =
  | { kind: 'folder-request'; folderId?: string }
  | { kind: 'flush-request'; requestId: number }
  | { kind: 'listing'; frame: import('./listing').ListingFrame }
  | { kind: 'worker-restarted' }
  | { kind: 'worker-stopped'; reason: string };
export interface DesktopBridge {
  version: 2;
  loadLast(): Promise<{ token: string } | null>;
  openFolder(folderId?: string): Promise<{ token: string } | null>;
  openDropped(paths: string[]): Promise<{ token: string; selected: PhotoId[] } | null>;
  recentFolders(): Promise<FolderRef[]>;
  pathsForFiles(files: File[]): string[];
  requestWorkerPort(): Promise<void>;
  updateUnsaved(edits: UnsavedSummary[]): void;
  respondFlush(requestId: number, unsaved: UnsavedSummary[]): void;
  onEvent(listener: (event: BridgeEvent) => void): () => void;
}
declare global { interface Window { xveon: DesktopBridge } }
