import * as ort from 'onnxruntime-web';
import type { CfaType, ModelSize } from '@/lib/types';

export interface ModelMeta {
  epoch?: number;
  base_width?: number;
  hl_head?: boolean;
  param_count?: number;
  size_mb?: number;
  dtype?: string;
  file?: string;
  train_psnr?: number;
  val_psnr?: number;
  val_hl_psnr?: number;
  train_loss?: number;
  val_loss?: number;
}

type Manifest = Record<string, ModelMeta>;

interface ModelEntry {
  session: ort.InferenceSession;
  meta: ModelMeta;
}

// Sessions keyed by manifest key (e.g. "xtrans_w16_base")
const sessions = new Map<string, ModelEntry>();
// Currently active model per CFA type
const active = new Map<CfaType, string>();

let manifest: Manifest = {};
let backend: string | null = null;
let initPromise: Promise<void> | null = null;
let currentSize: ModelSize = 'S';
let gpuDevice: GPUDevice | null = null;

const CHECKPOINTS_DIR = './checkpoints';

const SIZE_TO_WIDTH: Record<ModelSize, number> = { S: 16, M: 32, L: 64 };

async function fetchManifest(manifestUrl: string): Promise<Manifest> {
  let res: Response;
  try {
    res = await fetch(manifestUrl);
  } catch (e) {
    throw new Error(`Couldn't download the model list (${(e as Error).message}).`);
  }
  if (!res.ok) throw new Error(`Couldn't download the model list (HTTP ${res.status}).`);
  const manifest = await res.json() as Manifest;
  if (!manifest || Object.keys(manifest).length === 0) throw new Error('The model list is empty.');
  return manifest;
}

async function createSession(modelUrl: string): Promise<ort.InferenceSession> {
  // Try WebGPU first
  try {
    const session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ['webgpu'],
      preferredOutputLocation: 'gpu-buffer',
    });
    if (!backend) {
      backend = 'webgpu';
      console.log('ONNX Runtime: using WebGPU backend');
    }
    // Capture ORT's GPU device so our compute shaders can share buffers with it
    if (!gpuDevice) {
      gpuDevice = await ort.env.webgpu.device as GPUDevice;
      console.log('ORT WebGPU device captured for buffer interop');
    }
    return session;
  } catch (e) {
    console.warn('WebGPU not available:', (e as Error).message);
  }

  // Fall back to WASM
  try {
    const session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ['wasm'],
    });
    if (!backend) {
      backend = 'wasm';
      console.log('ONNX Runtime: using WASM backend (fallback)');
    }
    return session;
  } catch (e2) {
    console.warn('Multi-threaded WASM failed, trying single-threaded:', (e2 as Error).message);
    ort.env.wasm.numThreads = 1;
    const session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ['wasm'],
    });
    if (!backend) {
      backend = 'wasm (single-threaded)';
      console.log('ONNX Runtime: using WASM backend (single-threaded fallback)');
    }
    return session;
  }
}

/** Find the best manifest key for a given CFA type and base width.
 *  Prefers _hl variant, falls back to _base. */
function resolveModelKey(cfaType: CfaType, width: number): string | null {
  const prefix = cfaType === 'xtrans' ? 'xtrans' : 'bayer';
  // Prefer hl, then base
  for (const suffix of ['hl', 'base']) {
    const key = `${prefix}_w${width}_${suffix}`;
    if (manifest[key]) return key;
  }
  return null;
}

/** Get or load a session for a manifest key. */
async function getOrLoadSession(key: string): Promise<ModelEntry> {
  const existing = sessions.get(key);
  if (existing) return existing;

  const meta = manifest[key] ?? {};
  const modelUrl = `${CHECKPOINTS_DIR}/${meta.file ?? `${key}.onnx`}`;
  const session = await createSession(modelUrl);
  const entry = { session, meta };
  sessions.set(key, entry);
  console.log(`Loaded model ${key}: epoch ${meta.epoch ?? '?'}, PSNR ${meta.val_psnr ?? '?'} dB`);
  return entry;
}

/** Check which model sizes are available in the manifest for a CFA type. */
function getAvailableSizes(cfaType: CfaType): Set<ModelSize> {
  const sizes = new Set<ModelSize>();
  for (const [size, width] of Object.entries(SIZE_TO_WIDTH) as [ModelSize, number][]) {
    if (resolveModelKey(cfaType, width)) sizes.add(size);
  }
  return sizes;
}

async function initModels(size: ModelSize = 'S'): Promise<void> {
  if (initPromise) return initPromise;

  currentSize = size;
  initPromise = (async () => {
    // One WASM thread. Multithreading needs cross-origin isolation, which GitHub Pages does not
    // provide, and under isolation ORT's pthread workers try to load the app bundle and hang
    // (observed 2026-09-04). Revisit only with a verified isolated setup.
    ort.env.wasm.numThreads = 1;
    manifest = await fetchManifest(`${CHECKPOINTS_DIR}/models.json`);

    const width = SIZE_TO_WIDTH[size];
    for (const cfaType of ['xtrans', 'bayer'] as CfaType[]) {
      const key = resolveModelKey(cfaType, width);
      if (!key) {
        console.warn(`No ${size} model for ${cfaType}`);
        continue;
      }
      await getOrLoadSession(key);
      active.set(cfaType, key);
    }
  })();

  return initPromise;
}

/** Switch to a different model size. Returns once the new models are loaded. */
async function switchModelSize(size: ModelSize): Promise<void> {
  if (size === currentSize) return;
  currentSize = size;
  const width = SIZE_TO_WIDTH[size];

  for (const cfaType of ['xtrans', 'bayer'] as CfaType[]) {
    const key = resolveModelKey(cfaType, width);
    if (!key) continue;
    await getOrLoadSession(key);
    active.set(cfaType, key);
  }
}

/**
 * GPU-resident batch inference: GPUBuffer in → GPUBuffer out.
 * The input buffer must be created on the same device shared via ort.env.webgpu.device
 * with usage STORAGE | COPY_SRC. The returned GPUBuffer is owned by the caller
 * (dispose the tensor to release it when done with accumulation).
 */
async function runBatchGpu(
  cfaType: CfaType, inputBuffer: GPUBuffer, batchSize: number, patchSize: number,
): Promise<{ buffer: GPUBuffer; dispose: () => void }> {
  const key = active.get(cfaType);
  if (!key) throw new Error(`No active model for ${cfaType}`);
  const entry = sessions.get(key);
  if (!entry) throw new Error(`ONNX session not loaded for ${key}`);

  const inputTensor = ort.Tensor.fromGpuBuffer(inputBuffer, {
    dataType: 'float32' as const,
    dims: [batchSize, 5, patchSize, patchSize],
  });
  const results = await entry.session.run({ input: inputTensor });
  const outTensor = results.output;
  return {
    buffer: outTensor.gpuBuffer as GPUBuffer,
    dispose: () => outTensor.dispose(),
  };
}

export interface ModelRegistry {
  init(size: ModelSize): Promise<void>;
  switchSize(size: ModelSize): Promise<void>;
  availableSizes(cfaType: CfaType): Set<ModelSize>;
  runBatchGpu(
    cfaType: CfaType, inputBuffer: GPUBuffer, batchSize: number, patchSize: number,
  ): Promise<{ buffer: GPUBuffer; dispose: () => void }>;
  /** 'webgpu', 'wasm', 'wasm (single-threaded)' or null before init. */
  readonly backend: string | null;
  /** ORT's WebGPU device for buffer interop; null on the WASM backend. */
  readonly device: GPUDevice | null;
}

/** The loaded ONNX models. One registry per page: ONNX Runtime is a singleton. */
export const models: ModelRegistry = {
  init: initModels,
  switchSize: switchModelSize,
  availableSizes: getAvailableSizes,
  runBatchGpu,
  get backend() { return backend; },
  get device() { return gpuDevice; },
};
