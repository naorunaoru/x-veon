import type { DemosaicMethod } from '@/lib/types';
import { neuralNetStrategy } from './nn';
import { bilinearStrategy } from './bilinear-gpu';
import { dhtStrategy } from './dht-gpu';
import { poolStrategy } from './wasm-pool';
import type { DemosaicStrategy } from './strategy';

export type { DemosaicInput, DemosaicOutput, DemosaicStrategy, TraditionalMethod } from './strategy';
export { destroyDemosaicPool } from './wasm-pool';

/** One strategy per catalogue method (lib/catalog.ts). index.test.ts holds the two lists equal. */
export const DEMOSAIC_STRATEGIES: readonly DemosaicStrategy[] = [
  neuralNetStrategy,
  poolStrategy('markesteijn3'),
  poolStrategy('markesteijn1'),
  dhtStrategy,
  poolStrategy('ahd'),
  poolStrategy('ppg'),
  poolStrategy('mhc'),
  bilinearStrategy,
];

export function strategyFor(method: DemosaicMethod): DemosaicStrategy {
  const strategy = DEMOSAIC_STRATEGIES.find((candidate) => candidate.id === method);
  if (!strategy) throw new Error(`unknown demosaic method: ${method}`);
  return strategy;
}
