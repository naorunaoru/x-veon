import type { CfaType, DemosaicMethod, ExportFormat } from '@/lib/types';
import {
  methodKey,
  exportKey,
  type AdapterInfo,
  type GoldenBaseline,
  type GoldenMode,
  type Expectation,
} from './golden-compare';
export const SAMPLE_CONTRACT = [
  { file: 'DSCF3332.RAF', cfa: 'xtrans', traditional: 'dht' },
  { file: 'sony_a6400_21.arw', cfa: 'bayer', traditional: 'ahd' },
] as const;
export const TRADITIONAL: Record<CfaType, DemosaicMethod[]> = {
  xtrans: ['markesteijn3', 'markesteijn1', 'dht', 'bilinear'],
  bayer: ['ahd', 'ppg', 'mhc', 'bilinear'],
};
export const FORMATS: ExportFormat[] = ['jpeg-hdr', 'avif', 'tiff'];
export function expectationFor(mode: GoldenMode): Expectation {
  const keys: string[] = [],
    exportKeys: string[] = [];
  for (const sample of SAMPLE_CONTRACT) {
    keys.push(methodKey(sample.file, 'neural-net', 'S'));
    for (const method of mode === 'quick'
      ? [sample.traditional]
      : TRADITIONAL[sample.cfa])
      keys.push(methodKey(sample.file, method));
    if (mode === 'full')
      for (const format of FORMATS)
        exportKeys.push(exportKey(sample.file, format));
  }
  return { keys, exportKeys, runs: mode === 'quick' ? 2 : 1 };
}
export function selectBaseline(
  adapter: AdapterInfo,
  baselines: GoldenBaseline[],
): GoldenBaseline | null {
  if (
    !adapter.vendor ||
    !adapter.architecture ||
    adapter.vendor === 'unknown' ||
    adapter.architecture === 'unknown'
  )
    throw Error('Unknown GPU adapter');
  const matches = baselines.filter(
    (b) =>
      b.adapter.vendor === adapter.vendor &&
      b.adapter.architecture === adapter.architecture,
  );
  if (matches.length > 1) throw Error('Ambiguous GPU baseline');
  return matches[0] ?? null;
}
