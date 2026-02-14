import * as ort from 'onnxruntime-web';
import type { CfaType } from './types';

export interface ModelMeta {
  epoch?: string;
  best_val_psnr?: string;
  [key: string]: string | undefined;
}

interface ModelEntry {
  session: ort.InferenceSession;
  meta: ModelMeta;
}

const models = new Map<CfaType, ModelEntry>();
let backend: string | null = null;
let initPromise: Promise<void> | null = null;

async function fetchMeta(modelUrl: string): Promise<ModelMeta> {
  const metaUrl = modelUrl.replace(/\.onnx$/, '.meta.json');
  try {
    const res = await fetch(metaUrl);
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

async function loadModel(cfaType: CfaType, modelUrl: string): Promise<void> {
  const [session, meta] = await Promise.all([
    createSession(modelUrl),
    fetchMeta(modelUrl),
  ]);
  models.set(cfaType, { session, meta });
  console.log(`Loaded ${cfaType} model: epoch ${meta.epoch ?? '?'}, PSNR ${meta.best_val_psnr ?? '?'} dB`);
}

export async function initModels(): Promise<void> {
  if (initPromise) return initPromise;

  initPromise = (async () => {
    ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;

    // Load both models (first model determines backend)
    await loadModel('xtrans', './xtrans.onnx');
    await loadModel('bayer', './bayer.onnx');
  })();

  return initPromise;
}

export async function runTile(cfaType: CfaType, inputData: Float32Array, patchSize: number): Promise<Float32Array> {
  const entry = models.get(cfaType);
  if (!entry) throw new Error(`ONNX session not loaded for ${cfaType}`);
  const tensor = new ort.Tensor('float32', inputData, [1, 4, patchSize, patchSize]);
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
