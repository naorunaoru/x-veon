import * as ort from 'onnxruntime-web';
import type { CfaType, ModelSize } from './types';

export interface ModelMeta {
  epoch?: number;
  base_width?: number;
  cfa_type?: CfaType;
  checkpoint_version?: string;
  registry_status?: 'stable' | 'beta';
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

// Sessions keyed by manifest key (e.g. "xtrans-v6.1.4")
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
  try {
    const res = await fetch(manifestUrl);
    if (!res.ok) return {};
    return await res.json();
  } catch {
    return {};
  }
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

function versionSortKey(version?: string): [number, number, number, number] {
  const m = version?.match(/^v(\d+)\.(\d+)\.(\d+)(?:-w(\d+))?$/);
  if (!m) return [-1, -1, -1, -1];
  return [Number(m[1]), Number(m[2]), Number(m[3]), Number(m[4] ?? 0)];
}

function compareVersionKey(a: [number, number, number, number], b: [number, number, number, number]): number {
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return a[i] - b[i];
  }
  return 0;
}

/** Find the best manifest key for a given CFA type and base width.
 *  Prefers stable entries, then newest version. */
function resolveModelKey(cfaType: CfaType, width: number): string | null {
  let bestKey: string | null = null;
  let bestStable = -1;
  let bestVersion: [number, number, number, number] = [-1, -1, -1, -1];

  for (const [key, meta] of Object.entries(manifest)) {
    if ((meta.cfa_type ?? null) !== cfaType) continue;
    if ((meta.base_width ?? 16) !== width) continue;

    const stableScore = meta.registry_status === 'stable' ? 1 : 0;
    const versionKey = versionSortKey(meta.checkpoint_version);
    const better =
      stableScore > bestStable ||
      (stableScore === bestStable && compareVersionKey(versionKey, bestVersion) > 0);
    if (better) {
      bestStable = stableScore;
      bestVersion = versionKey;
      bestKey = key;
    }
  }
  return bestKey;
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
export function getAvailableSizes(cfaType: CfaType): Set<ModelSize> {
  const sizes = new Set<ModelSize>();
  for (const [size, width] of Object.entries(SIZE_TO_WIDTH) as [ModelSize, number][]) {
    if (resolveModelKey(cfaType, width)) sizes.add(size);
  }
  return sizes;
}

export async function initModels(size: ModelSize = 'S'): Promise<void> {
  if (initPromise) return initPromise;

  currentSize = size;
  initPromise = (async () => {
    ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
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
export async function switchModelSize(size: ModelSize): Promise<void> {
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

export async function runBatch(
  cfaType: CfaType, batchInput: Float32Array, batchSize: number, patchSize: number,
  wb: [number, number, number] = [1, 1, 1],
): Promise<Float32Array> {
  const key = active.get(cfaType);
  if (!key) throw new Error(`No active model for ${cfaType}`);
  const entry = sessions.get(key);
  if (!entry) throw new Error(`ONNX session not loaded for ${key}`);
  const tensor = new ort.Tensor('float32', batchInput, [batchSize, 1, patchSize, patchSize]);
  // WB coefficients: repeat per batch element
  const wbData = new Float32Array(batchSize * 3);
  for (let i = 0; i < batchSize; i++) { wbData[i * 3] = wb[0]; wbData[i * 3 + 1] = wb[1]; wbData[i * 3 + 2] = wb[2]; }
  const wbTensor = new ort.Tensor('float32', wbData, [batchSize, 3]);
  const results = await entry.session.run({ input: tensor, wb: wbTensor });
  return results.output.data as Float32Array;
}

/**
 * GPU-resident batch inference: GPUBuffer in → GPUBuffer out.
 * The input buffer must be created on the same device shared via ort.env.webgpu.device
 * with usage STORAGE | COPY_SRC. The returned GPUBuffer is owned by the caller
 * (dispose the tensor to release it when done with accumulation).
 */
export async function runBatchGpu(
  cfaType: CfaType, inputBuffer: GPUBuffer, batchSize: number, patchSize: number,
  wb: [number, number, number] = [1, 1, 1],
): Promise<{ buffer: GPUBuffer; dispose: () => void }> {
  const key = active.get(cfaType);
  if (!key) throw new Error(`No active model for ${cfaType}`);
  const entry = sessions.get(key);
  if (!entry) throw new Error(`ONNX session not loaded for ${key}`);

  const inputTensor = ort.Tensor.fromGpuBuffer(inputBuffer, {
    dataType: 'float32' as const,
    dims: [batchSize, 1, patchSize, patchSize],
  });
  // WB coefficients: CPU tensor (small, not worth GPU upload)
  const wbData = new Float32Array(batchSize * 3);
  for (let i = 0; i < batchSize; i++) { wbData[i * 3] = wb[0]; wbData[i * 3 + 1] = wb[1]; wbData[i * 3 + 2] = wb[2]; }
  const wbTensor = new ort.Tensor('float32', wbData, [batchSize, 3]);
  const results = await entry.session.run({ input: inputTensor, wb: wbTensor });
  const outTensor = results.output;
  return {
    buffer: outTensor.gpuBuffer as GPUBuffer,
    dispose: () => outTensor.dispose(),
  };
}

/** Get ORT's GPUDevice for compute shader interop (null if WASM backend). */
export function getInferenceDevice(): GPUDevice | null {
  return gpuDevice;
}

export function getBackend(): string | null {
  return backend;
}

export function getModelMeta(cfaType?: CfaType): ModelMeta {
  if (cfaType) {
    const key = active.get(cfaType);
    if (key) return sessions.get(key)?.meta ?? {};
  }
  return sessions.get(active.get('xtrans') ?? '')?.meta ?? {};
}

export function getCurrentModelSize(): ModelSize {
  return currentSize;
}
