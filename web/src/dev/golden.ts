/**
 * Browser-only golden harness. It drives the real application through the store and hashes
 * renderer readbacks for a fixed pair of representative RAW files. Normal builds exclude it.
 */
import { configFromPreset, configWithOverrides, computeTonescaleParams, deriveHdrConfig } from '@/renderer/grading/opendrt-params';
import { BUILD } from '@/lib/channel';
import { encoderFor } from '@/pipeline/export';
import { switchModelSize } from '@/pipeline/inference';
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
  type Expectation,
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
const HDR_PEAK_LUMINANCE = 1000;
const SAMPLE_CONTRACT: readonly ManifestSample[] = [
  { file: 'DSCF3332.RAF', cfa: 'xtrans', traditional: 'dht' },
  { file: 'sony_a6400_21.arw', cfa: 'bayer', traditional: 'ahd' },
];
const TRADITIONAL: Record<CfaType, DemosaicMethod[]> = {
  xtrans: ['markesteijn3', 'markesteijn1', 'dht', 'bilinear'],
  bayer: ['ahd', 'ppg', 'mhc', 'bilinear'],
};
const FORMATS: ExportFormat[] = ['jpeg-hdr', 'avif', 'tiff'];

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
      if (previous.rendererRef === null && state.rendererRef !== null) finish();
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
  const renderer = useAppStore.getState().rendererRef;
  if (!renderer) throw new Error('renderer not published');

  const scene = await hashFloat32(await renderer.readbackImage());
  const base = configFromPreset('default');
  const config = configWithOverrides(base, {}, {});
  const display = await hashFloat32(
    await renderer.renderForExport(config, computeTonescaleParams(config), 'rec709'),
  );
  const darkConfig = configWithOverrides(base, {}, { exposure: -4 });
  const displayDark = await hashFloat32(
    await renderer.renderForExport(darkConfig, computeTonescaleParams(darkConfig), 'rec709'),
  );
  return { scene, display, displayDark };
}

/** Mirrors useExport until Plan B introduces an export service. */
async function exportOnce(id: string, format: ExportFormat): Promise<{ bytes: number; sha256: string }> {
  const state = useAppStore.getState();
  const file = fileOf(id);
  const renderer = state.rendererRef;
  if (!file?.result || !renderer) throw new Error('nothing to export');

  const { exportData } = file.result;
  const baseConfig = configFromPreset(file.lookPreset);
  const sdrConfig = configWithOverrides(
    baseConfig,
    file.openDrtOverrides,
    file.preProcessOverrides,
  );
  const sdrTonescale = computeTonescaleParams(sdrConfig);
  let data: Float32Array;
  let hdrData: Float32Array | null = null;
  let peakLuminance = sdrConfig.peak_luminance;

  if (format === 'jpeg-hdr' || format === 'avif') {
    const hdrConfig = deriveHdrConfig(sdrConfig, HDR_PEAK_LUMINANCE);
    const hdrTonescale = computeTonescaleParams(hdrConfig);
    peakLuminance = HDR_PEAK_LUMINANCE;
    if (format === 'jpeg-hdr') {
      data = await renderer.renderForExport(sdrConfig, sdrTonescale, 'rec709');
      hdrData = await renderer.renderForExport(hdrConfig, hdrTonescale, 'rec2020');
    } else {
      data = await renderer.renderForExport(hdrConfig, hdrTonescale, 'rec2020');
    }
  } else {
    data = await renderer.renderForExport(sdrConfig, sdrTonescale, 'rec709');
  }

  const blob = await encoderFor(format).encode(
    data, hdrData, exportData.width, exportData.height, exportData.orientation, state.exportQuality, peakLuminance,
  );
  const bytes = new Uint8Array(await blob.arrayBuffer());
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
): Promise<void> {
  const getStore = useAppStore.getState;
  getStore().setDemosaicMethod('neural-net');
  getStore().setModelSize(size);
  await switchModelSize(size);
  const file = await fetchSample(sample.file);
  const previousIds = new Set(getStore().files.map((entry) => entry.id));

  let rendererRefresh = armRendererRefresh();
  const startedAt = performance.now();
  try {
    getStore().addFiles([file]);
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
        getStore().setDemosaicMethod(method);
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
      getStore().removeFile(id);
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

function expectationFor(mode: GoldenMode): Expectation {
  const keys: string[] = [];
  const exportKeys: string[] = [];
  for (const sample of SAMPLE_CONTRACT) {
    keys.push(methodKey(sample.file, 'neural-net', 'S'));
    const traditional = mode === 'full' ? TRADITIONAL[sample.cfa] : [sample.traditional];
    for (const method of traditional) keys.push(methodKey(sample.file, method));
    if (mode === 'full') {
      for (const format of FORMATS) exportKeys.push(exportKey(sample.file, format));
    }
  }
  return { keys, exportKeys, runs: mode === 'full' ? 1 : 2 };
}

async function adapterInfo(): Promise<AdapterInfo> {
  const adapter = await navigator.gpu?.requestAdapter();
  const info = adapter?.info;
  return {
    vendor: info?.vendor ?? 'unknown',
    architecture: info?.architecture ?? 'unknown',
  };
}

function loadBaseline(): GoldenBaseline | null {
  const modules = import.meta.glob('../test/golden/baseline.json', {
    eager: true,
    import: 'default',
  }) as Record<string, GoldenBaseline>;
  return Object.values(modules)[0] ?? null;
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

export async function runGolden(): Promise<void> {
  const mode: GoldenMode = new URLSearchParams(location.search).get('golden') === 'full'
    ? 'full'
    : 'quick';
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
        + '(copy the sample RAWs into web/public/samples and write the manifest)',
      );
    }
    const manifest = await manifestResponse.json() as Manifest;
    validateManifest(manifest);

    const expected = expectationFor(mode);
    const passes = mode === 'full' ? 1 : 2;
    for (let run = 1; run <= passes; run += 1) {
      for (const sample of SAMPLE_CONTRACT) {
        const traditional = mode === 'full' ? TRADITIONAL[sample.cfa] : [sample.traditional];
        await cycle(sample, 'S', traditional, mode === 'full', run, runs, exports);
      }
    }

    const report = buildReport(
      mode,
      runs,
      exports,
      await adapterInfo(),
      BUILD.sha,
      new Date().toISOString(),
    );
    const results = compareToBaseline(report, loadBaseline(), expected);
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
