import init, { encode_image } from '../../wasm/encoder/pkg/xtrans_encoder_wasm.js';

let ready = false;

self.onmessage = async (e: MessageEvent) => {
  if (e.data.type !== 'encode') return;

  try {
    if (!ready) {
      await init();
      ready = true;
    }

    const { data, hdrData, width, height, orientation, format, quality, peakLuminance } = e.data;
    const result = encode_image(
      new Float32Array(data),
      new Float32Array(hdrData),
      width, height,
      orientation, format, quality,
      peakLuminance,
    );
    self.postMessage({ type: 'done', data: result.buffer }, [result.buffer] as any);
  } catch (err: unknown) {
    self.postMessage({ type: 'error', message: err instanceof Error ? err.message : String(err) });
  }
};
