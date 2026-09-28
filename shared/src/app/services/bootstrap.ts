import { useAppStore } from '@/app/store';
import { initPipeline } from '@/pipeline';
import { setPipeline } from '@/app/services/processing';
import { restore } from '@/app/services/persistence';
import { matchLensFor } from '@/app/services/library';
import { getHost } from './host';

/** Initialise the pipeline and restore the library; `signal.cancelled` stops the store writes. */
export async function initApp(signal: { cancelled: boolean }): Promise<void> {
  const store = useAppStore.getState;
  try {
    const [ctx, restored] = await Promise.all([initPipeline({ modelSize: 'S' }), restore()]);
    if (signal.cancelled) {
      getHost().library.release?.(restored.files);
      return;
    }
    setPipeline(ctx);

    // Merges with anything imported while startup ran, and restores settings even for an empty library.
    store().restoreFromDb(restored.files, restored.settings);

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
    for (const file of restored.files) matchLensFor(file.id);


  } catch (e) {
    if (!signal.cancelled) store().setInitError((e as Error).message);
  }
}
