import * as ort from 'onnxruntime-web';
import type { CfaType } from './types';

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

const models = new Map<CfaType, ModelEntry>();
let backend: string | null = null;
let initPromise: Promise<void> | null = null;

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

const CHECKPOINTS_DIR = './checkpoints';
const MODEL_KEYS: Record<CfaType, string> = {
  xtrans: 'xtrans_w16_base',
  bayer: 'bayer_w32_hl',
};

export async function initModels(): Promise<void> {
  if (initPromise) return initPromise;

  initPromise = (async () => {
    ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;

    const manifest = await fetchManifest(`${CHECKPOINTS_DIR}/models.json`);

    for (const [cfaType, key] of Object.entries(MODEL_KEYS) as [CfaType, string][]) {
      const meta = manifest[key] ?? {};
      const modelUrl = `${CHECKPOINTS_DIR}/${meta.file ?? `${key}.onnx`}`;
      const session = await createSession(modelUrl);
      models.set(cfaType, { session, meta });
      console.log(`Loaded ${cfaType} model: epoch ${meta.epoch ?? '?'}, PSNR ${meta.val_psnr ?? '?'} dB`);
    }
  })();

  return initPromise;
}

export async function runBatch(
  cfaType: CfaType, batchInput: Float32Array, batchSize: number, patchSize: number,
): Promise<Float32Array> {
  const entry = models.get(cfaType);
  if (!entry) throw new Error(`ONNX session not loaded for ${cfaType}`);
  const tensor = new ort.Tensor('float32', batchInput, [batchSize, 5, patchSize, patchSize]);
  const results = await entry.session.run({ input: tensor });
  return results.output.data as Float32Array;
}

export function getBackend(): string | null {
  return backend;
}

export function getModelMeta(cfaType?: CfaType): ModelMeta {
  if (cfaType) {
    return models.get(cfaType)?.meta ?? {};
  }
  // Default: return xtrans meta for backward compat
  return models.get('xtrans')?.meta ?? {};
}
