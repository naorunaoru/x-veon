import init, { encode_image } from '../../wasm/encoder/pkg/xtrans_encoder_wasm.js';

let ready = false;

self.onmessage = async (e: MessageEvent) => {
  if (e.data.type !== 'encode') return;

  try {
    if (!ready) {
      await init();
      ready = true;
    }

    const { hwc, width, height, xyzToCam, wbCoeffs, orientation, format, quality, odrtConfig } = e.data;
    const result = encode_image(
      new Float32Array(hwc),
      width, height,
      new Float32Array(xyzToCam),
      new Float32Array(wbCoeffs),
      orientation, format, quality,
      new Float32Array(odrtConfig),
    );
    self.postMessage({ type: 'done', data: result.buffer }, [result.buffer] as any);
  } catch (err: unknown) {
    self.postMessage({ type: 'error', message: err instanceof Error ? err.message : String(err) });
  }
};
