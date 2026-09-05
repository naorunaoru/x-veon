import { useEffect, useRef } from 'react';
import { useAppStore } from '@/app/store';
import { useProcessing } from '@/app/hooks/useProcessing';
import type { FileStatus } from '@/app/store';
import { isMethodValidForCfa } from '@/lib/catalog';

/** Pure decision: should the given file be auto-processed right now? */
export function shouldAutoProcess(
  file: { status: FileStatus } | undefined,
  initialized: boolean,
  isProcessing: boolean,
): boolean {
  return initialized && !isProcessing && file?.status === 'queued';
}

/**
 * Owns the processing side-effects previously hosted in SettingsPanel:
 *  - auto-process the selected file when it is queued (fresh drop / restore),
 *  - reprocess the selected file when the demosaic method changes,
 *  - fall back to neural-net if the current method is invalid for the file's CFA.
 * Mount once, near the app root.
 */
export function useAutoProcess(): void {
  const initialized = useAppStore((s) => s.initialized);
  const demosaicMethod = useAppStore((s) => s.demosaicMethod);
  const setDemosaicMethod = useAppStore((s) => s.setDemosaicMethod);
  const selectedFile = useAppStore((s) => s.files.find((f) => f.id === s.selectedFileId));
  const { processFile, isProcessing } = useProcessing();

  // CFA-aware method fallback: X-Trans-only / Bayer-only methods can't run on the other CFA.
  const cfaType = selectedFile?.cfaType ?? null;
  useEffect(() => {
    if (!cfaType) return;
    if (!isMethodValidForCfa(demosaicMethod, cfaType)) {
      setDemosaicMethod('neural-net');
    }
  }, [cfaType, demosaicMethod, setDemosaicMethod]);

  // Auto-process queued selected file.
  useEffect(() => {
    if (shouldAutoProcess(selectedFile, initialized, isProcessing)) {
      processFile(selectedFile!.id);
    }
  }, [selectedFile?.id, selectedFile?.status, initialized, isProcessing, processFile, selectedFile]);

  // Reprocess on method change.
  const prevMethodRef = useRef(demosaicMethod);
  useEffect(() => {
    if (prevMethodRef.current === demosaicMethod) return;
    prevMethodRef.current = demosaicMethod;
    if (
      initialized && selectedFile && !isProcessing &&
      (selectedFile.status === 'done' || selectedFile.status === 'error')
    ) {
      processFile(selectedFile.id);
    }
  }, [demosaicMethod, initialized, selectedFile, isProcessing, processFile]);
}
