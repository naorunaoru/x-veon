import { useAppStore } from '@/app/store';
import { initPipeline } from '@/pipeline';
import { setPipeline } from '@/app/services/processing';
import {
  restore, flushPersistence, unsavedEdits, onUnsavedChange, type UnsavedEdit,
} from '@/app/services/persistence';
import { matchLensFor, openFolder, folderSwitchVersion } from '@/app/services/library';
import { getHost } from './host';

/** Initialise the pipeline and restore the library; `signal.cancelled` stops the store writes. */
export async function initApp(signal: { cancelled: boolean }): Promise<void> {
  const store = useAppStore.getState;
  const generation = folderSwitchVersion();
  try {
    const [ctx, restored] = await Promise.all([initPipeline({ modelSize: 'S' }), restore()]);
    if (signal.cancelled) {
      getHost().library.release?.(restored.files);
      return;
    }
    setPipeline(ctx);

    // Merges with anything imported while startup ran, and restores settings even for an empty library.
    const settings = { ...restored.settings };
    // Older betas saved sizes they never restored. Ignore defaults no longer shipped.
    if (
      settings.modelSize &&
      !ctx.models.availableSizes('xtrans').has(settings.modelSize) &&
      !ctx.models.availableSizes('bayer').has(settings.modelSize)
    )
      delete settings.modelSize;
    const overtaken = restored.folder !== undefined && generation !== folderSwitchVersion();
    const restoredFiles = overtaken ? [] : restored.files;
    if (overtaken) getHost().library.release?.(restored.files);
    store().restoreFromDb(restoredFiles, settings);
    if (!overtaken && restored.folder !== undefined) useAppStore.setState({ folder: restored.folder });

    const backend = ctx.models.backend ?? 'unknown';

    // Probe display HDR (headroom via Window Management API / screen API)
    const hdrDisplayInfo = await getHost().display.probe();
    if (signal.cancelled) return;
    if (hdrDisplayInfo.supported) {
      store().setDisplayHdr(true, hdrDisplayInfo.headroom);
      // If headroom is a conservative fallback and the Window Management API
      // could provide an accurate value, prompt the user for permission
      if (!hdrDisplayInfo.accurate && getHost().display.requestAccurateHeadroom) {
        store().setHdrPermissionNeeded(true);
      }
    }

    store().setInitialized(backend);

    // Match lenses for restored files that have metadata but no profile yet
    for (const file of restoredFiles) matchLensFor(file.id);
  } catch (e) {
    if (!signal.cancelled) store().setInitError((e as Error).message);
  }
}

/** Register optional host controls for this mounted app and release them together. */
export function startHostCoordination(): () => void {
  const library = getHost().library;
  const report = (entries: UnsavedEdit[]) => library.reportUnsaved?.(
    entries.map(({ id, name, folder, error }) => ({ id, name, folder, error })),
  );
  const stopFolder = library.onFolderRequest?.(folder => {
    void openFolder(folder).catch(error => console.warn('Folder open failed:', error));
  });
  const stopFlush = library.onFlushRequest?.(flushPersistence);
  const stopUnsaved = library.reportUnsaved ? onUnsavedChange(report) : undefined;
  report(unsavedEdits());
  return () => {
    stopFolder?.();
    stopFlush?.();
    stopUnsaved?.();
  };
}
