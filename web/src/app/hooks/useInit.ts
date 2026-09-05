import { useEffect } from 'react';
import { useAppStore } from '@/app/store';
import type { QueuedFile } from '@/app/store';
import { initPipeline } from '@/pipeline';
import { setPipeline } from '@/app/services/processing';
import { probeHdrDisplay, hasWindowManagementApi } from '@/renderer/hdr-display';
import { getAllFiles, getSetting } from '@/app/storage/idb-storage';
import type { PersistedFile } from '@/app/storage/idb-storage';
import { listRawFileIds, deleteAllForFile, readThumbnail } from '@/app/storage/opfs-storage';
import type { DemosaicMethod, ExportFormat } from '@/lib/types';
import { deserializeResultMeta } from '@/lib/types';
import type { OpenDrtConfig, PreProcessConfig } from '@/renderer/grading/opendrt-params';
import { matchLens } from '@/app/lens/lensfun';

async function persistedToQueued(p: PersistedFile): Promise<QueuedFile> {
  // Load thumbnail from OPFS
  const thumbBlob = await readThumbnail(p.id).catch(() => null);

  return {
    id: p.id,
    file: null,
    name: p.name,
    originalName: p.originalName,
    thumbnailUrl: thumbBlob ? URL.createObjectURL(thumbBlob) : null,
    metadata: p.camera ? {
      camera: p.camera,
      lensModel: p.lensModel ?? '',
      focalLength: p.focalLength ?? 0,
      fNumber: p.fNumber ?? 0,
    } : null,
    cfaType: p.cfaType,
    status: p.status === 'done' && p.resultMeta ? 'done' : 'queued',
    error: null,
    progress: null,
    result: p.resultMeta ? deserializeResultMeta(p.resultMeta) : null,
    resultMethod: p.resultMethod,
    lensProfile: p.lensProfile ?? null,
    lookPreset: p.lookPreset,
    openDrtOverrides: p.openDrtOverrides as Partial<OpenDrtConfig>,
    preProcessOverrides: (p.preProcessOverrides ?? {}) as Partial<PreProcessConfig>,
  };
}

export function useInit() {
  const setInitialized = useAppStore((s) => s.setInitialized);
  const setInitError = useAppStore((s) => s.setInitError);

  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        // Initialize WASM, models, GPU demosaic in parallel
        // Restore from IndexedDB concurrently
        const [ctx, persistedFiles, demosaicMethod, exportFormat, exportQuality, selectedFileId] =
          await Promise.all([
            initPipeline({ modelSize: 'S' }),
            getAllFiles().catch(() => [] as PersistedFile[]),
            getSetting<DemosaicMethod>('demosaicMethod').catch(() => undefined),
            getSetting<ExportFormat>('exportFormat').catch(() => undefined),
            getSetting<number>('exportQuality').catch(() => undefined),
            getSetting<string | null>('selectedFileId').catch(() => undefined),
          ]);

        if (cancelled) return;
        setPipeline(ctx);

        const files: QueuedFile[] = [];
        for (const p of persistedFiles) {
          files.push(await persistedToQueued(p));
        }

        // Restore state into Zustand
        if (files.length > 0) {
          useAppStore.getState().restoreFromDb(files, {
            demosaicMethod,
            exportFormat,
            exportQuality,
            selectedFileId: selectedFileId ?? undefined,
          });
        }

        const backend = ctx.models.backend ?? 'unknown';

        // Probe display HDR (headroom via Window Management API / screen API)
        const hdrDisplayInfo = await probeHdrDisplay();
        if (hdrDisplayInfo.supported) {
          useAppStore.getState().setDisplayHdr(true, hdrDisplayInfo.headroom);
          // If headroom is a conservative fallback and the Window Management API
          // could provide an accurate value, prompt the user for permission
          if (!hdrDisplayInfo.accurate && hasWindowManagementApi()) {
            useAppStore.getState().setHdrPermissionNeeded(true);
          }
        }

        setInitialized(backend);

        // Match lenses for restored files that have metadata but no profile yet
        for (const qf of files) {
          if (!qf.lensProfile && qf.metadata?.lensModel) {
            matchLens(qf.metadata.camera, qf.metadata.lensModel)
              .then((profile) => {
                if (profile) useAppStore.getState().setFileLensProfile(qf.id, profile);
              })
              .catch((e) => console.warn('Lens match failed:', e));
          }
        }

        // Orphan cleanup: remove OPFS entries not in IDB (fire-and-forget)
        cleanupOrphans(new Set(files.map((f) => f.id)));

        // Request persistent storage (best-effort)
        navigator.storage?.persist?.().catch(() => {});
      } catch (e) {
        if (!cancelled) setInitError((e as Error).message);
      }
    }

    init();
    return () => {
      cancelled = true;
    };
  }, [setInitialized, setInitError]);
}

async function cleanupOrphans(knownIds: Set<string>): Promise<void> {
  try {
    const rawIds = await listRawFileIds();
    for (const id of rawIds) {
      if (!knownIds.has(id)) {
        deleteAllForFile(id).catch(() => {});
      }
    }
  } catch {
    // Non-critical — silently ignore
  }
}
