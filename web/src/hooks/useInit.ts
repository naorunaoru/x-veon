import { useEffect } from 'react';
import { useAppStore } from '@/store';
import type { QueuedFile } from '@/store';
import { initWasm } from '@/pipeline/raf-decoder';
import { initModels, getBackend, getModelMeta } from '@/pipeline/inference';
import { initDemosaicGpuSafe } from '@/pipeline/demosaic';
import { probeHdrDisplay } from '@/gl/hdr-display';
import { getAllFiles, getSetting } from '@/lib/idb-storage';
import type { PersistedFile } from '@/lib/idb-storage';
import { hasHwc, hwcKey, listRawFileIds, listHwcFileIds, deleteAllForFile, readThumbnail } from '@/lib/opfs-storage';
import { deserializeResultMeta } from '@/pipeline/types';
import type { DemosaicMethod, ExportFormat, ProcessingResultMeta } from '@/pipeline/types';
import type { OpenDrtConfig } from '@/gl/opendrt-params';

async function persistedToQueued(p: PersistedFile): Promise<QueuedFile> {
  const result: ProcessingResultMeta | null = p.resultMeta
    ? deserializeResultMeta(p.resultMeta)
    : null;

  // Load thumbnail from OPFS
  const thumbBlob = await readThumbnail(p.id).catch(() => null);

  return {
    id: p.id,
    file: null,
    name: p.name,
    originalName: p.originalName,
    thumbnailUrl: thumbBlob ? URL.createObjectURL(thumbBlob) : null,
    metadata: p.camera ? { camera: p.camera } : null,
    cfaType: p.cfaType,
    status: p.status === 'done' ? 'done' : 'queued',
    error: p.status === 'error' ? p.error : null,
    progress: null,
    result,
    resultMethod: p.resultMethod,
    cachedResults: result && p.resultMethod
      ? { [p.resultMethod]: result } as Partial<Record<DemosaicMethod, ProcessingResultMeta>>
      : {},
    lookPreset: p.lookPreset,
    openDrtOverrides: p.openDrtOverrides as Partial<OpenDrtConfig>,
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
        const [,, , persistedFiles, demosaicMethod, exportFormat, exportQuality, selectedFileId] =
          await Promise.all([
            initWasm(),
            initModels(),
            initDemosaicGpuSafe(),
            getAllFiles().catch(() => [] as PersistedFile[]),
            getSetting<DemosaicMethod>('demosaicMethod').catch(() => undefined),
            getSetting<ExportFormat>('exportFormat').catch(() => undefined),
            getSetting<number>('exportQuality').catch(() => undefined),
            getSetting<string | null>('selectedFileId').catch(() => undefined),
          ]);

        if (cancelled) return;

        // Validate restored 'done' files — check HWC exists in OPFS
        const files: QueuedFile[] = [];
        for (const p of persistedFiles) {
          const qf = await persistedToQueued(p);
          if (qf.status === 'done' && qf.resultMethod) {
            const exists = await hasHwc(hwcKey(qf.id, qf.resultMethod));
            if (!exists) {
              qf.status = 'queued';
              qf.result = null;
              qf.resultMethod = null;
              qf.cachedResults = {};
            }
          }
          files.push(qf);
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

        const backend = getBackend() ?? 'unknown';

        // Probe display HDR (headroom via Window Management API / screen API)
        const hdrDisplayInfo = await probeHdrDisplay();
        if (hdrDisplayInfo.supported) {
          useAppStore.getState().setDisplayHdr(true, hdrDisplayInfo.headroom);
        }

        setInitialized(backend, getModelMeta());

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
    const [rawIds, hwcIds] = await Promise.all([listRawFileIds(), listHwcFileIds()]);
    const allOpfsIds = new Set([...rawIds, ...hwcIds]);
    for (const id of allOpfsIds) {
      if (!knownIds.has(id)) {
        deleteAllForFile(id).catch(() => {});
      }
    }
  } catch {
    // Non-critical — silently ignore
  }
}
