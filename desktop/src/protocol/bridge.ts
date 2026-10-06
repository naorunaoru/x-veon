import type { ExportFormat } from '@/lib/types';
import type { DisplayReadings, FolderRef, NewerRelease, PhotoId, UnsavedSummary } from '@/host';
export type { UnsavedSummary } from '@/host';
export type BridgeEvent =
  | { kind: 'folder-request'; folderId?: string }
  | { kind: 'flush-request'; requestId: number }
  | { kind: 'listing'; frame: import('./listing').ListingFrame }
  | { kind: 'worker-restarted'; worker?: string }
  | { kind: 'worker-stopped'; reason: string };
export interface DesktopBridge {
  version: 2;
  chooseExportDestination(photoId: PhotoId, format: ExportFormat): Promise<{ token: string; name: string } | null>;
  revealExport(token: string): Promise<void>;
  loadLast(): Promise<{ token: string } | null>;
  openFolder(folderId?: string): Promise<{ token: string } | null>;
  openDropped(files: File[]): Promise<{ token: string; selected: PhotoId[] } | null>;
  recentFolders(): Promise<FolderRef[]>;
  displayReadings(): Promise<DisplayReadings | null>;
  checkForUpdate(): Promise<NewerRelease | null>;
  requestWorkerPort(requestId: string): Promise<void>;
  updateUnsaved(edits: UnsavedSummary[]): void;
  respondFlush(requestId: number, unsaved: UnsavedSummary[]): void;
  onEvent(listener: (event: BridgeEvent) => void): () => void;
}
declare global { interface Window { xveon: DesktopBridge } }
