import init, { encode_image } from '../wasm-encoder/pkg/xtrans_encoder_wasm.js';

let ready = false;

self.onmessage = async (e) => {
  if (e.data.type !== 'encode') return;

  try {
    if (!ready) {
      await init();
      ready = true;
    }

    const { hwc, width, height, xyzToCam, orientation, format, quality } = e.data;
    const result = encode_image(
      new Float32Array(hwc),
      width, height,
      new Float32Array(xyzToCam),
      orientation, format, quality,
    );
    self.postMessage({ type: 'done', data: result.buffer }, [result.buffer]);
  } catch (err) {
    self.postMessage({ type: 'error', message: err.message || String(err) });
  }
};
