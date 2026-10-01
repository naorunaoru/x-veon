export type { UnsavedSummary } from '@/host';
export type BridgeEvent =
  | { kind: 'folder-request'; folderId?: string }
  | { kind: 'flush-request'; requestId: number }
  | { kind: 'listing'; frame: import('./listing').ListingFrame }
  | { kind: 'worker-restarted' }
  | { kind: 'worker-stopped'; reason: string };
export type ReportName =
  | 'golden'
  | 'timing'
  | 'transport'
  | 'capabilities'
  | 'error';
export interface SpikeBridge {
  version: 1;
  environment: { sandboxed: boolean; contextIsolated: boolean };
  request(kind: 'connect' | 'restart' | 'diagnostics'): Promise<unknown>;
  report(name: ReportName, value: unknown): Promise<void>;
}
declare global {
  interface Window {
    xveon: SpikeBridge;
  }
}
