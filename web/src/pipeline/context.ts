import type { DemosaicMethod, ModelSize } from '@/lib/types';
import type { ModelRegistry } from './inference';

/** Everything a processing run needs that outlives the run: the shared device and the loaded models. */
export interface PipelineContext {
  device: GPUDevice;
  models: ModelRegistry;
}

export interface ProcessOptions {
  method: DemosaicMethod;
  /** Requested model size; the caller makes sure it is loaded (see ModelRegistry.switchSize). */
  modelSize: ModelSize;
  /** Tiles done / total for the neural-net strategy. Unused by the UI today. */
  onProgress?: (done: number, total: number) => void;
}
