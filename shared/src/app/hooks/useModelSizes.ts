import { useCallback } from 'react';
import { models } from '@/pipeline/inference';
import type { CfaType, ModelSize, ModelIdentity } from '@/lib/types';

/** Model sizes available for the selected file's sensor, and a way to switch the loaded model. */
export function useModelSizes(cfaType: CfaType | null): {
  available: Set<ModelSize>;
  switchTo: (size: ModelSize) => Promise<void>;
  modelFor: (size: ModelSize) => ModelIdentity;
} {
  const available = cfaType ? models.availableSizes(cfaType) : new Set<ModelSize>(['S']);
  const switchTo = useCallback((size: ModelSize) => models.switchSize(size), []);
  return { available, switchTo, modelFor: size => { if (!cfaType) throw new Error('Select a photo first.'); return models.resolve(cfaType, { size, sha256: '' }, size).model; } };
}
