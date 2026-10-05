import { getDevice } from '@/gpu/device';
import { useAppStore } from '@/app/store';
import { importFiles } from '@/app/services/library';
import { renderExport } from '@/app/services/export';
import { models } from '@/pipeline/inference';
import { BUILD } from '@/lib/channel';
import { BENCH_SAMPLE } from './golden-contract';
import { hashBytes } from './golden-compare';
export interface BenchRun { encodeMs: number | null; totalMs: number; bytes: number; sha256: string }
export interface BenchReport {
  status: 'BENCH'; sample: string; width: number; height: number; commit: string;
  adapter: { vendor: string; architecture: string }; runs: BenchRun[];
  median: { encodeMs: number | null; totalMs: number };
}
export function median(values: number[]): number {
  if (!values.length || values.some(value => !Number.isFinite(value))) throw Error('Invalid median samples');
  const sorted = [...values].sort((a, b) => a - b), middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}
function waitForPhoto(id: string): Promise<void> {
  return new Promise((resolve, reject) => {
    let unsubscribe = () => {};
    const timer = setTimeout(() => { unsubscribe(); reject(Error('Benchmark photo processing timed out')); }, 240_000);
    const check = () => {
      const file = useAppStore.getState().files.find(file => file.id === id);
      if (file?.status !== 'done' && file?.status !== 'error' && file) return;
      clearTimeout(timer); unsubscribe();
      if (file?.status === 'done') resolve();
      else reject(Error(file?.error || 'Benchmark photo unavailable'));
    };
    unsubscribe = useAppStore.subscribe(check); check();
  });
}
export async function runBench(): Promise<BenchReport> {
  const store = useAppStore.getState;
  store().selectFile(null); store().setDemosaicMethod('neural-net'); store().setModelSize('S'); store().setExportQuality(95);
  await models.switchSize('S');
  const response = await fetch(`${import.meta.env.BASE_URL}samples/${BENCH_SAMPLE}`);
  if (!response.ok) throw Error(`sample ${BENCH_SAMPLE}: HTTP ${response.status}`);
  const previous = new Set(store().files.map(file => file.id));
  await importFiles([new File([await response.blob()], BENCH_SAMPLE)]);
  const file = store().files.find(file => !previous.has(file.id));
  if (!file) throw Error('Benchmark import did not create a photo');
  store().selectFile(file.id); await waitForPhoto(file.id);
  const data = store().files.find(candidate => candidate.id === file.id)?.result?.exportData;
  if (!data) throw Error('Benchmark result unavailable');
  const runs: BenchRun[] = [];
  for (let index = 0; index < 4; index++) {
    const started = performance.now();
    const result = await renderExport(file.id, 'avif', 95);
    const totalMs = performance.now() - started;
    const bytes = result.blob ? new Uint8Array(await result.blob.arrayBuffer()) : undefined;
    const sha256 = result.sha256 ?? (bytes ? await hashBytes(bytes) : '');
    const byteLength = result.bytes ?? bytes?.byteLength;
    if (!/^[a-f0-9]{64}$/.test(sha256) || !Number.isSafeInteger(byteLength) || byteLength! <= 0)
      throw Error('Invalid benchmark export receipt');
    if (index) runs.push({ encodeMs: result.encodeMs ?? null, totalMs, bytes: byteLength!, sha256 });
  }
  const info = (await getDevice()).adapterInfo;
  const rotated = data.orientation === 'Rotate90' || data.orientation === 'Rotate270';
  return {
    status: 'BENCH', sample: BENCH_SAMPLE, width: rotated ? data.height : data.width,
    height: rotated ? data.width : data.height, commit: BUILD.sha,
    adapter: { vendor: info?.vendor ?? 'unknown', architecture: info?.architecture ?? 'unknown' }, runs,
    median: { encodeMs: runs.every(run => run.encodeMs !== null) ? median(runs.map(run => run.encodeMs!)) : null,
      totalMs: median(runs.map(run => run.totalMs)) },
  };
}
