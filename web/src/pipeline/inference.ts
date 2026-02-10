import * as ort from 'onnxruntime-web';

let session: ort.InferenceSession | null = null;
let backend: string | null = null;
let initPromise: Promise<void> | null = null;

export async function initModel(modelUrl: string): Promise<void> {
  // Guard against double-init (React strict mode)
  if (initPromise) return initPromise;

  initPromise = (async () => {
    ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;

    // Try WebGPU first
    try {
      session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ['webgpu'],
      });
      backend = 'webgpu';
      console.log('ONNX Runtime: using WebGPU backend');
      return;
    } catch (e) {
      console.warn('WebGPU not available:', (e as Error).message);
    }

    // Fall back to WASM
    try {
      session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ['wasm'],
      });
      backend = 'wasm';
      console.log('ONNX Runtime: using WASM backend (fallback)');
    } catch (e) {
      console.warn('Multi-threaded WASM failed, trying single-threaded:', (e as Error).message);
      ort.env.wasm.numThreads = 1;
      session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ['wasm'],
      });
      backend = 'wasm (single-threaded)';
      console.log('ONNX Runtime: using WASM backend (single-threaded fallback)');
    }
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
