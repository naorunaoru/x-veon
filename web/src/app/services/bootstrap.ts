import { useAppStore } from '@/app/store';
import { initPipeline } from '@/pipeline';
import { setPipeline } from '@/app/services/processing';
import { restore } from '@/app/services/persistence';
import { matchLensFor, cleanupOrphans } from '@/app/services/library';
import { probeHdrDisplay, hasWindowManagementApi } from '@/renderer/hdr-display';

/** Initialise the pipeline and restore the library; `signal.cancelled` stops the store writes. */
export async function initApp(signal: { cancelled: boolean }): Promise<void> {
  const store = useAppStore.getState;
  try {
    const [ctx, restored] = await Promise.all([initPipeline({ modelSize: 'S' }), restore()]);
    if (signal.cancelled) {
      for (const file of restored.files) {
        if (file.thumbnailUrl) URL.revokeObjectURL(file.thumbnailUrl);
      }
      return;
    }
    setPipeline(ctx);

    if (restored.files.length > 0) store().restoreFromDb(restored.files, restored.settings);

    const backend = ctx.models.backend ?? 'unknown';

    // Probe display HDR (headroom via Window Management API / screen API)
    const hdrDisplayInfo = await probeHdrDisplay();
    if (signal.cancelled) return;
    if (hdrDisplayInfo.supported) {
      store().setDisplayHdr(true, hdrDisplayInfo.headroom);
      // If headroom is a conservative fallback and the Window Management API
      // could provide an accurate value, prompt the user for permission
      if (!hdrDisplayInfo.accurate && hasWindowManagementApi()) {
        store().setHdrPermissionNeeded(true);
      }
    }

    store().setInitialized(backend);

    // Match lenses for restored files that have metadata but no profile yet
    for (const file of restored.files) matchLensFor(file.id);

    // Orphan cleanup: remove OPFS entries not in IDB (fire-and-forget)
    cleanupOrphans(new Set(restored.files.map((f) => f.id)));

    // Request persistent storage (best-effort)
    navigator.storage?.persist?.().catch(() => {});
  } catch (e) {
    if (!signal.cancelled) store().setInitError((e as Error).message);
  }
}
