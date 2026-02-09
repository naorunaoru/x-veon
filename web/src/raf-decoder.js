let wasmModule = null;

/**
 * Initialize the WASM module (rawloader-wasm).
 */
export async function initWasm() {
  wasmModule = await import('../wasm/pkg/rawloader_wasm.js');
  await wasmModule.default();
}

/**
 * Decode a RAF file from an ArrayBuffer.
 *
 * @param {ArrayBuffer} arrayBuffer
 * @returns {object} Decoded image with raw data and metadata
 */
export function decodeRaf(arrayBuffer) {
  if (!wasmModule) throw new Error('WASM not initialized');

  const bytes = new Uint8Array(arrayBuffer);
  const img = wasmModule.decode_image(bytes);

  return {
    data: img.get_data(),               // Uint16Array [H * W]
    width: img.get_width(),
    height: img.get_height(),
    wbCoeffs: img.get_wb_coeffs(),      // Float32Array [R, G, B, E]
    blackLevels: img.get_blacklevels(), // Uint16Array [4]
    whiteLevels: img.get_whitelevels(), // Uint16Array [4]
    xyzToCam: img.get_xyz_to_cam(),     // Float32Array [9] (3x3 row-major)
    orientation: img.get_orientation(),
    make: img.get_make(),
    model: img.get_model(),
    cfaStr: img.get_cfastr(),
    cfaWidth: img.get_cfawidth(),
    crops: img.get_crops(),             // Uint16Array [top, right, bottom, left]
  };
}
