import * as ort from 'onnxruntime-web';

let session = null;
let backend = null;

/**
 * Initialize the ONNX Runtime session.
 * Tries WebGPU first, falls back to WASM.
 *
 * @param {string} modelUrl - URL/path to the .onnx model file
 */
export async function initModel(modelUrl) {
  // Configure ONNX Runtime WASM backend paths
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
    console.warn('WebGPU not available:', e.message);
  }

  // Fall back to WASM
  try {
    session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ['wasm'],
    });
    backend = 'wasm';
    console.log('ONNX Runtime: using WASM backend (fallback)');
  } catch (e) {
    // If multi-threaded WASM fails (no SharedArrayBuffer), try single-threaded
    console.warn('Multi-threaded WASM failed, trying single-threaded:', e.message);
    ort.env.wasm.numThreads = 1;
    session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ['wasm'],
    });
    backend = 'wasm (single-threaded)';
    console.log('ONNX Runtime: using WASM backend (single-threaded fallback)');
  }
}

/**
 * Run inference on a single tile.
 *
 * @param {Float32Array} inputData - Flat array of shape [4, patchSize, patchSize]
 * @param {number} patchSize
 * @returns {Promise<Float32Array>} - Output of shape [3, patchSize, patchSize]
 */
export async function runTile(inputData, patchSize) {
  const tensor = new ort.Tensor('float32', inputData, [1, 4, patchSize, patchSize]);
  const results = await session.run({ input: tensor });
  return results.output.data;
}

/**
 * @returns {string|null} The active backend name ('webgpu' or 'wasm')
 */
export function getBackend() {
  return backend;
}
