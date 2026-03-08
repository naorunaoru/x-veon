import * as ort from 'onnxruntime-web';
import type { CfaType, ModelSize } from './types';

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
    });
    if (!backend) {
      backend = 'webgpu';
      console.log('ONNX Runtime: using WebGPU backend');
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
): Promise<Float32Array> {
  const key = active.get(cfaType);
  if (!key) throw new Error(`No active model for ${cfaType}`);
  const entry = sessions.get(key);
  if (!entry) throw new Error(`ONNX session not loaded for ${key}`);
  const tensor = new ort.Tensor('float32', batchInput, [batchSize, 5, patchSize, patchSize]);
  const results = await entry.session.run({ input: tensor });
  return results.output.data as Float32Array;
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
