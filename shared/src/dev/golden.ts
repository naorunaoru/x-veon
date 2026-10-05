import { getDevice } from '@/gpu/device';
import { SAMPLE_CONTRACT, TRADITIONAL, FORMATS, expectationFor, selectBaseline } from './golden-contract';
import { importFiles, removeFile } from '@/app/services/library';
/**
 * Host-neutral golden harness. It drives the real application through the store and hashes
 * renderer readbacks for a fixed pair of representative RAW files. Normal builds exclude it.
 */
import { configFromPreset, configWithOverrides, computeTonescaleParams } from '@/renderer/grading/opendrt-params';
import { BUILD } from '@/lib/channel';
import { renderExport } from '@/app/services/export';
import { models } from '@/pipeline/inference';
import type { CfaType, DemosaicMethod, ExportFormat, ModelSize } from '@/lib/types';
import { useAppStore } from '@/app/store';
import {
  buildReport,
  compareToBaseline,
  exportKey,
  hashBytes,
  hashFloat32,
  methodKey,
  overallStatus,
  type AdapterInfo,
  type ExportRun,
  type GoldenBaseline,
  type GoldenMode,
  type RunEntry,
} from './golden-compare';

interface ManifestSample {
  file: string;
  cfa: CfaType;
  traditional: DemosaicMethod;
}

interface Manifest {
  samples: ManifestSample[];
}

const STEP_TIMEOUT_MS = 240_000;

type State = ReturnType<typeof useAppStore.getState>;

function waitForState(predicate: (state: State) => boolean, label: string): Promise<void> {
  return new Promise((resolve, reject) => {
    if (predicate(useAppStore.getState())) {
      resolve();
      return;
    }

    const timer = setTimeout(() => {
      unsubscribe();
      reject(new Error(`timeout waiting for ${label}`));
    }, STEP_TIMEOUT_MS);
    const unsubscribe = useAppStore.subscribe((state) => {
      if (!predicate(state)) return;
      clearTimeout(timer);
      unsubscribe();
      resolve();
    });
  });
}

/** Arm before processing so a fast null-to-renderer transition cannot be missed. */
function armRendererRefresh(): { promise: Promise<void>; cancel: () => void } {
  let cancel = () => {};
  const promise = new Promise<void>((resolve, reject) => {
    let settled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let unsubscribe = () => {};
    const finish = (error?: Error) => {
      if (settled) return;
      settled = true;
      if (timer) clearTimeout(timer);
      unsubscribe();
      if (error) reject(error);
      else resolve();
    };

    cancel = () => finish();
    timer = setTimeout(
      () => finish(new Error('timeout waiting for renderer refresh')),
      STEP_TIMEOUT_MS,
    );
    unsubscribe = useAppStore.subscribe((state, previous) => {
      if (previous.renderer === null && state.renderer !== null) finish();
    });
  });

  // The processing-state waiter is awaited first. Mark this promise handled immediately in
  // case both timers expire together; awaiting the original promise still propagates errors.
  void promise.catch(() => {});
  return { promise, cancel };
}

const fileOf = (id: string) => useAppStore.getState().files.find((file) => file.id === id);
const statusOf = (id: string) => fileOf(id)?.status;

async function readbackHashes(): Promise<{ scene: string; display: string; displayDark: string }> {
  const renderer = useAppStore.getState().renderer;
  if (!renderer) throw new Error('renderer not published');

  const scene = await hashFloat32(await renderer.readbackImage());
  const base = configFromPreset('default');
  const config = configWithOverrides(base, {}, {});
  const display = await hashFloat32(
    await renderer.readback(config, computeTonescaleParams(config), 'rec709'),
  );
  const darkConfig = configWithOverrides(base, {}, { exposure: -4 });
  const displayDark = await hashFloat32(
    await renderer.readback(darkConfig, computeTonescaleParams(darkConfig), 'rec709'),
  );
  return { scene, display, displayDark };
}

export async function exportOnce(id: string, format: ExportFormat): Promise<{ bytes: number; sha256: string }> {
  const result = await renderExport(id, format, useAppStore.getState().exportQuality);
  if (result.sha256 !== undefined) {
    if (!/^[a-f0-9]{64}$/.test(result.sha256) || !Number.isSafeInteger(result.bytes) || result.bytes! <= 0)
      throw new Error('The exporter returned an invalid golden receipt.');
    return { bytes: result.bytes!, sha256: result.sha256 };
  }
  if (!result.blob) throw new Error('The exporter returned no bytes for the golden check.');
  const bytes = new Uint8Array(await result.blob.arrayBuffer());
  return { bytes: bytes.byteLength, sha256: await hashBytes(bytes) };
}

async function fetchSample(name: string): Promise<File> {
  const response = await fetch(`${import.meta.env.BASE_URL}samples/${name}`);
  if (!response.ok) throw new Error(`sample ${name}: HTTP ${response.status}`);
  return new File([await response.blob()], name);
}

async function cycle(
  sample: ManifestSample,
  size: ModelSize,
  traditional: DemosaicMethod[],
  exportsWanted: boolean,
  run: number,
  runs: RunEntry[],
  exports: ExportRun[],
  cleanup: (id: string) => Promise<void>,
): Promise<void> {
  const getStore = useAppStore.getState;
  getStore().setDemosaicMethod('neural-net');
  getStore().setModelSize(size);
  await models.switchSize(size);
  const file = await fetchSample(sample.file);
  const previousIds = new Set(getStore().files.map((entry) => entry.id));

  let rendererRefresh = armRendererRefresh();
  const startedAt = performance.now();
  try {
    await importFiles([file]);
    const entry = getStore().files.find((candidate) => !previousIds.has(candidate.id));
    if (!entry) throw new Error('addFiles did not create an entry');
    const id = entry.id;
    getStore().selectFile(id);
    const neuralKey = methodKey(sample.file, 'neural-net', size);

    try {
      await waitForState(
        () => statusOf(id) === 'done' || statusOf(id) === 'error',
        `${neuralKey} done`,
      );
      if (statusOf(id) === 'error') {
        const error = fileOf(id)?.error ?? 'error';
        runs.push({
          key: neuralKey, run, scene: '', display: '', displayDark: '', elapsedMs: 0, error,
        });
        for (const method of traditional) {
          runs.push({
            key: methodKey(sample.file, method),
            run,
            scene: '',
            display: '',
            displayDark: '',
            elapsedMs: 0,
            error: `skipped: ${error}`,
          });
        }
        if (exportsWanted) {
          for (const format of FORMATS) {
            exports.push({
              key: exportKey(sample.file, format),
              bytes: 0,
              sha256: '',
              error: `skipped: ${error}`,
            });
          }
        }
        return;
      }

      await rendererRefresh.promise;
      runs.push({
        key: neuralKey,
        run,
        ...(await readbackHashes()),
        elapsedMs: Math.round(performance.now() - startedAt),
      });

      if (exportsWanted) {
        for (const format of FORMATS) {
          try {
            exports.push({ key: exportKey(sample.file, format), ...(await exportOnce(id, format)) });
          } catch (error) {
            exports.push({
              key: exportKey(sample.file, format),
              bytes: 0,
              sha256: '',
              error: error instanceof Error ? error.message : String(error),
            });
          }
        }
      }

      for (const method of traditional) {
        const key = methodKey(sample.file, method);
        rendererRefresh = armRendererRefresh();
        const methodStartedAt = performance.now();
        getStore().setFileDemosaicMethod(id, method);
        await waitForState(() => statusOf(id) === 'processing', `${key} start`);
        await waitForState(
          () => statusOf(id) === 'done' || statusOf(id) === 'error',
          `${key} done`,
        );
        if (statusOf(id) === 'error') {
          runs.push({
            key,
            run,
            scene: '',
            display: '',
            displayDark: '',
            elapsedMs: 0,
            error: fileOf(id)?.error ?? 'error',
          });
          rendererRefresh.cancel();
          continue;
        }

      await rendererRefresh.promise;
        runs.push({
          key,
          run,
          ...(await readbackHashes()),
          elapsedMs: Math.round(performance.now() - methodStartedAt),
        });
      }
    } finally {
      await cleanup(id);
      getStore().selectFile(null);
      getStore().setDemosaicMethod('neural-net');
    }
  } finally {
    rendererRefresh.cancel();
  }
}

function validateManifest(manifest: Manifest): void {
  if (JSON.stringify(manifest.samples) !== JSON.stringify(SAMPLE_CONTRACT)) {
    throw new Error(
      `sample manifest must exactly match the golden contract: ${JSON.stringify(SAMPLE_CONTRACT)}`,
    );
  }
}

async function adapterInfo(): Promise<AdapterInfo> {
  const info = (await getDevice()).adapterInfo;
  return { vendor: info?.vendor ?? 'unknown', architecture: info?.architecture ?? 'unknown' };
}
function loadBaseline(adapter: AdapterInfo): GoldenBaseline | null {
  const modules = import.meta.glob('../test/golden/baselines/*.json', {eager:true,import:'default'}) as Record<string, GoldenBaseline>;
  return selectBaseline(adapter, Object.values(modules));
}

function publish(payload: unknown, status: string): void {
  (window as unknown as { __golden: unknown }).__golden = payload;
  document.title = `golden: ${status}`;
  const report = document.createElement('pre');
  report.id = 'golden-report';
  report.style.cssText = 'position:fixed;left:0;right:0;bottom:0;max-height:45vh;overflow:auto;margin:0;padding:12px;background:rgba(0,0,0,.85);color:#7f7;font:11px/1.4 monospace;z-index:9999;white-space:pre-wrap';
  report.textContent = JSON.stringify(payload, null, 2);
  document.body.appendChild(report);
  console.log('[golden]', JSON.stringify(payload));
}

export async function runGolden(cleanup: (id: string) => Promise<void> = removeFile, opts: { encoder?: 'wasm' | 'native' } = {}): Promise<void> {
  const requested = new URLSearchParams(location.search).get('golden');
  const mode: GoldenMode = requested === 'full' ? 'full' : requested === 'render' ? 'render' : 'quick';
  document.title = `golden: running (${mode})`;
  const runs: RunEntry[] = [];
  const exports: ExportRun[] = [];

  try {
    await waitForState((state) => state.initialized, 'app initialised');
    useAppStore.getState().setExportQuality(95);
    useAppStore.getState().selectFile(null);

    const manifestResponse = await fetch(`${import.meta.env.BASE_URL}samples/manifest.json`);
    if (!manifestResponse.ok) {
      throw new Error(
        `samples/manifest.json: HTTP ${manifestResponse.status} `
        + '(copy the sample RAWs into shared/public/samples and write the manifest)',
      );
    }
    const manifest = await manifestResponse.json() as Manifest;
    validateManifest(manifest);

    const expected = expectationFor(mode);
    const passes = mode === 'quick' ? 2 : 1;
    for (let run = 1; run <= passes; run += 1) {
      for (const sample of SAMPLE_CONTRACT) {
        const traditional = mode === 'quick' ? [sample.traditional] : TRADITIONAL[sample.cfa];
        await cycle(sample, 'S', traditional, mode === 'full', run, runs, exports, cleanup);
      }
    }

    const report = buildReport(
      mode,
      runs,
      exports,
      await adapterInfo(),
      BUILD.sha,
      new Date().toISOString(),
      opts.encoder ?? 'wasm',
    );
    const results = compareToBaseline(report, loadBaseline(report.adapter), expected);
    const status = overallStatus(results);
    publish({ status, results, report, expected }, status);
  } catch (error) {
    publish({
      status: 'ERROR',
      error: error instanceof Error ? error.message : String(error),
      runs,
      exports,
    }, 'ERROR');
  }
}
