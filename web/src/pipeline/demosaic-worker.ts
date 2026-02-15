import init, { demosaic_image, demosaic_bayer } from '../../wasm-demosaic/pkg/demosaic_wasm.js';

let ready = false;

self.onmessage = async (e: MessageEvent) => {
  if (e.data.type !== 'demosaic') return;

  try {
    if (!ready) {
      await init();
      ready = true;
    }

    const { cfa, width, height, dy, dx, algorithm, bayerVariant } = e.data;
    const cfaData = new Float32Array(cfa);

    let result: Float32Array;
    if (bayerVariant) {
      // Bayer path: uses dedicated demosaic_bayer with algorithm selection
      result = demosaic_bayer(cfaData, width, height, bayerVariant, algorithm);
    } else {
      // X-Trans path
      result = demosaic_image(cfaData, width, height, dy, dx, algorithm);
    }

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
