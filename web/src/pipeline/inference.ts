import * as ort from 'onnxruntime-web';

export interface ModelMeta {
  epoch?: string;
  best_val_psnr?: string;
  [key: string]: string | undefined;
}

let session: ort.InferenceSession | null = null;
let backend: string | null = null;
let modelMeta: ModelMeta = {};
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

export async function initModel(modelUrl: string): Promise<void> {
  // Guard against double-init (React strict mode)
  if (initPromise) return initPromise;

  initPromise = (async () => {
    ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;

    // Fetch model metadata sidecar in parallel with session creation
    const metaPromise = fetchMeta(modelUrl);

    // Try WebGPU first
    try {
      session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ['webgpu'],
      });
      backend = 'webgpu';
      console.log('ONNX Runtime: using WebGPU backend');
    } catch (e) {
      console.warn('WebGPU not available:', (e as Error).message);

      // Fall back to WASM
      try {
        session = await ort.InferenceSession.create(modelUrl, {
          executionProviders: ['wasm'],
        });
        backend = 'wasm';
        console.log('ONNX Runtime: using WASM backend (fallback)');
      } catch (e2) {
        console.warn('Multi-threaded WASM failed, trying single-threaded:', (e2 as Error).message);
        ort.env.wasm.numThreads = 1;
        session = await ort.InferenceSession.create(modelUrl, {
          executionProviders: ['wasm'],
        });
        backend = 'wasm (single-threaded)';
        console.log('ONNX Runtime: using WASM backend (single-threaded fallback)');
      }
    }

    modelMeta = await metaPromise;
  })();

  return initPromise;
}

export async function runTile(inputData: Float32Array, patchSize: number): Promise<Float32Array> {
  if (!session) throw new Error('ONNX session not initialized');
  const tensor = new ort.Tensor('float32', inputData, [1, 4, patchSize, patchSize]);
  const results = await session.run({ input: tensor });
  return results.output.data as Float32Array;
}

export function getBackend(): string | null {
  return backend;
}

export function getModelMeta(): ModelMeta {
  return modelMeta;
}
