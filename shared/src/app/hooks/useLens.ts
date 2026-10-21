import { useCallback, useEffect, useMemo, useState } from 'react';
import { useAppStore } from '@/app/store';
import { lensPickKey } from '@/app/store/lens';
import { listLenses, type LensChoice } from '@/app/lens/lensfun';
import { lensFrameFor, resolvedLensFor } from '@/app/lens/remap';
import { lensReadouts, lensStatus, type LensReadouts, type ResolvedLens } from '@/app/lens/resolve';
import { ensureLensPick } from '@/app/services/lens-picks';
import { effectiveLensEdit, isLensEditModified, type LensEdit, type LensManual, type LensSource } from '@/lib/lens/edit';

const NO_LENSES: LensChoice[] = [];

/** The lens list per camera string; a panel that closes and opens again reads it here. A failed list is dropped, so the next open retries. */
const listed = new Map<string, Promise<LensChoice[]>>();
function lensesFor(camera: string): Promise<LensChoice[]> {
  let pending = listed.get(camera);
  if (!pending) {
    pending = listLenses(camera);
    listed.set(camera, pending);
    pending.catch(() => listed.delete(camera));
  }
  return pending;
}

/** Per-photo lens context for the Lens panel. Reads the selected file. */
export function useLens(): {
  fileId: string | null;
  /** The photo has decoded, so there is something to correct. */
  ready: boolean;
  edit: LensEdit;
  resolved: ResolvedLens | null;
  readouts: LensReadouts | null;
  status: { line: string; detail: string | null };
  manualLive: boolean;
  modified: boolean;
  /** Empty until the list for the selected photo's camera has arrived. */
  lenses: LensChoice[];
  setSource(source: LensSource): void;
  setToggle(key: 'distortion' | 'ca' | 'vignetting', on: boolean): void;
  setManual(key: keyof LensManual, value: number): void;
  reset(): void;
} {
  const file = useAppStore((s) => s.files.find((f) => f.id === s.selectedFileId));
  const picks = useAppStore((s) => s.lensPicks);
  const setFileLens = useAppStore((s) => s.setFileLens);
  const resetFileLens = useAppStore((s) => s.resetFileLens);

  const fileId = file?.id ?? null;
  const camera = file?.metadata?.camera ?? '';
  const edit = useMemo(() => effectiveLensEdit(file?.edit ?? {}), [file?.edit]);

  const { resolved, readouts } = useMemo(() => {
    const resolved = file ? resolvedLensFor(file, picks) : null;
    const frame = file ? lensFrameFor(file) : null;
    const halfDiagPx = frame && (frame.halfDiagPx ?? Math.hypot(frame.width, frame.height) / 2);
    return { resolved, readouts: resolved && halfDiagPx ? lensReadouts(resolved.curves, halfDiagPx) : null };
  }, [file, picks]);

  const exifLens = file?.metadata?.lensModel ?? '';
  // A hand-picked lens is named by its own profile once that is loaded, never by the automatic match.
  const lensfunModel = typeof edit.source === 'object'
    ? picks[lensPickKey(edit.source.lensfun.file, edit.source.lensfun.model)]?.lensModel ?? null
    : file?.lensProfile?.lensModel ?? null;
  const status = lensStatus({ resolved, edit, exifLens, lensfunModel, matching: !!exifLens && !file?.lensMatched });

  const [loaded, setLoaded] = useState<{ camera: string; choices: LensChoice[] }>({ camera: '', choices: NO_LENSES });
  useEffect(() => {
    if (!camera) return;
    let current = true;
    lensesFor(camera)
      .then((choices) => { if (current) setLoaded({ camera, choices }); })
      .catch((error) => console.warn('Lens list failed:', error));
    return () => { current = false; };
  }, [camera]);

  const setSource = useCallback((source: LensSource) => {
    if (!fileId) return;
    setFileLens(fileId, { source });
    if (typeof source === 'object') ensureLensPick(fileId, source.lensfun);
  }, [fileId, setFileLens]);
  const setToggle = useCallback((key: 'distortion' | 'ca' | 'vignetting', on: boolean) => {
    if (fileId) setFileLens(fileId, { [key]: on });
  }, [fileId, setFileLens]);
  // The store replaces `manual` wholesale, so a change carries the other four values along.
  const setManual = useCallback((key: keyof LensManual, value: number) => {
    if (fileId) setFileLens(fileId, { manual: { ...edit.manual, [key]: value } });
  }, [fileId, setFileLens, edit.manual]);
  const reset = useCallback(() => { if (fileId) resetFileLens(fileId); }, [fileId, resetFileLens]);

  return {
    fileId,
    ready: resolved !== null,
    edit,
    resolved,
    readouts,
    status,
    manualLive: resolved?.manualLive ?? false,
    modified: isLensEditModified(file?.edit ?? {}),
    lenses: loaded.camera === camera ? loaded.choices : NO_LENSES,
    setSource,
    setToggle,
    setManual,
    reset,
  };
}
