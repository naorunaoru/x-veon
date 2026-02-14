import init, { demosaic_image } from '../../wasm-demosaic-pkg/demosaic_wasm.js';

let ready = false;

self.onmessage = async (e: MessageEvent) => {
  if (e.data.type !== 'demosaic') return;

  try {
    if (!ready) {
      await init();
      ready = true;
    }

    const { cfa, width, height, dy, dx, algorithm } = e.data;
    const result = demosaic_image(
      new Float32Array(cfa),
      width, height, dy, dx, algorithm,
    );
    self.postMessage(
      { type: 'done', data: result.buffer },
      [result.buffer] as any,
    );
  } catch (err: unknown) {
    self.postMessage({
      type: 'error',
      message: err instanceof Error ? err.message : String(err),
    });
  }
};
