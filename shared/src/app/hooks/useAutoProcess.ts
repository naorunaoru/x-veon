import { useEffect } from 'react';
import { useAppStore } from '@/app/store';
import { useProcessing } from './useProcessing';
import { processingKey } from '@/app/store/photo';
import type { FileStatus } from '@/app/store';
import { isMethodValidForCfa } from '@/lib/catalog';
export function shouldAutoProcess(
  file: { status: FileStatus } | undefined,
  initialized: boolean,
  isProcessing: boolean,
): boolean {
  return initialized && !isProcessing && file?.status === 'queued';
}
export function useAutoProcess(): void {
  const state = useAppStore();
  const file = state.files.find((f) => f.id === state.selectedFileId);
  const { processFile, isProcessing } = useProcessing();
  const key = file ? processingKey(file, state) : null;
  useEffect(() => {
    if (!file || !state.initialized || isProcessing) return;
    const method = file.edit.demosaicMethod ?? state.demosaicMethod;
    if (file.cfaType && !isMethodValidForCfa(method, file.cfaType) && file.editing !== 'view-only') {
      state.setFileDemosaicMethod(file.id, 'neural-net');
      return;
    }
    if (file.status === 'queued' || (file.processedKey !== null && file.processedKey !== key))
      void processFile(file.id);
  }, [file?.id, file?.status, file?.processedKey, key, state.initialized, isProcessing, processFile]);
}
