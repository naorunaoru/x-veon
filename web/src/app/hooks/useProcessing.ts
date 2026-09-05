import { useAppStore } from '@/app/store';
import { processFile } from '@/app/services/processing';

/** The processing service for components: start a run, and whether one is in flight. */
export function useProcessing(): { processFile: (fileId: string) => Promise<void>; isProcessing: boolean } {
  const isProcessing = useAppStore((s) => s.processingFileId !== null);
  return { processFile, isProcessing };
}
