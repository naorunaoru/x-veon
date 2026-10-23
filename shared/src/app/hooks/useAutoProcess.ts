import { useEffect } from 'react';
import { useAppStore } from '@/app/store';
import { useProcessing } from './useProcessing';
import { needsProcessing, processingKey } from '@/app/store/photo';
import type { FileStatus } from '@/app/store';
import { isMethodValidForCfa } from '@/lib/catalog';
import { effectiveMethod } from '@/app/photo-edit';
export function shouldAutoProcess(
  file: { status: FileStatus } | undefined,
  initialized: boolean,
  isProcessing: boolean,
): boolean {
  return initialized && !isProcessing && file?.status === 'queued';
}
export function useAutoProcess(): void {
  const state = useAppStore();
  const file = state.files.find((f) => f.id === state.currentId);
  const { processFile, isProcessing } = useProcessing();
  const key = file ? processingKey(file, state) : null;
  useEffect(() => {
    if (!file || !state.initialized || isProcessing) return;
    const method = effectiveMethod(file.edit, state.demosaicMethod, file.cfaType);
    if (file.edit.demosaicMethod && file.cfaType && !isMethodValidForCfa(method, file.cfaType) && file.editing !== 'view-only') {
      state.setFileDemosaicMethod(file.id, 'neural-net');
      return;
    }
    if (needsProcessing(file, state)) void processFile(file.id);
  }, [file?.id, file?.status, file?.processedKey, key, state.initialized, isProcessing, processFile]);
}
