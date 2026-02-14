import { initDemosaicGpu, gpuAvailable, runBilinearGpu, runDhtGpu } from './demosaic-gpu';

export async function initDemosaicGpuSafe(): Promise<boolean> {
  return initDemosaicGpu();
}

let worker: Worker | null = null;

function getWorker(): Worker {
  if (!worker) {
    worker = new Worker(new URL('./demosaic-worker.ts', import.meta.url), { type: 'module' });
  }
  return worker;
}

function runDemosaicWorker(
  cfa: Float32Array,
  width: number,
  height: number,
  dy: number,
  dx: number,
  algorithm: 'bilinear' | 'markesteijn1',
): Promise<Float32Array> {
  return new Promise((resolve, reject) => {
    const w = getWorker();
    const cfaCopy = cfa.slice();

    w.onmessage = (e) => {
      if (e.data.type === 'done') {
        resolve(new Float32Array(e.data.data));
      } else if (e.data.type === 'error') {
        reject(new Error(e.data.message));
      }
    };
    w.onerror = (e) => reject(new Error(e.message));

    w.postMessage({
      type: 'demosaic',
      cfa: cfaCopy.buffer,
      width, height, dy, dx, algorithm,
    }, [cfaCopy.buffer]);
  });
}

export async function runDemosaic(
  cfa: Float32Array,
  width: number,
  height: number,
  dy: number,
  dx: number,
  algorithm: 'bilinear' | 'markesteijn1' | 'dht',
): Promise<Float32Array> {
  // GPU paths
  if (algorithm === 'bilinear' && gpuAvailable()) {
    try {
      console.time('[demosaic] gpu bilinear');
      const result = await runBilinearGpu(cfa, width, height, dy, dx);
      console.timeEnd('[demosaic] gpu bilinear');
      return result;
    } catch (e) {
      console.warn('[demosaic] GPU bilinear failed, falling back to worker:', e);
    }
  }

  if (algorithm === 'dht') {
    if (gpuAvailable()) {
      try {
        console.time('[demosaic] gpu dht');
        const result = await runDhtGpu(cfa, width, height, dy, dx);
        console.timeEnd('[demosaic] gpu dht');
        return result;
      } catch (e) {
        console.warn('[demosaic] GPU DHT failed, falling back to bilinear worker:', e);
      }
    } else {
      console.warn('[demosaic] DHT requires WebGPU, falling back to bilinear worker');
    }
    // DHT fallback: bilinear via worker
    const result = await runDemosaicWorker(cfa, width, height, dy, dx, 'bilinear');
    return result;
  }

  // Worker path (off main thread)
  console.time(`[demosaic] worker ${algorithm}`);
  const result = await runDemosaicWorker(cfa, width, height, dy, dx, algorithm);
  console.timeEnd(`[demosaic] worker ${algorithm}`);
  return result;
}
