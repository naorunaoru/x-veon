import type { DemosaicMethod, ModelSize, ModelIdentity } from '@/lib/types';
import type { ModelRegistry } from './inference';

/** Everything a processing run needs that outlives the run: the shared device and the loaded models. */
export interface PipelineContext {
  device: GPUDevice;
  models: ModelRegistry;
}

export interface ProcessOptions {
  method: DemosaicMethod;
  /** Resolve a Settings default against the decoded CFA; explicit photo methods stay explicit. */
  resolveDefault?: boolean;
  /** Requested model size; the caller makes sure it is loaded (see ModelRegistry.switchSize). */
  modelSize: ModelSize;
  model?: ModelIdentity | null;
  /** Optional diagnostic output: strategy execution through GPU completion, excluding model activation/postprocessing. */
  timings?: { demosaicMs?: number };
  /** Tiles done / total for the neural-net strategy. Unused by the UI today. */
  onProgress?: (done: number, total: number) => void;
}
