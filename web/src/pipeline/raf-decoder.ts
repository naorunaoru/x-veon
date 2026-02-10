import type { RawImage } from './types';

let wasmModule: Awaited<typeof import('../../wasm-pkg/rawloader_wasm.js')> | null = null;

export async function initWasm(): Promise<void> {
  wasmModule = await import('../../wasm-pkg/rawloader_wasm.js');
  await wasmModule.default();
}

export function decodeRaf(arrayBuffer: ArrayBuffer): RawImage {
  if (!wasmModule) throw new Error('WASM not initialized');

  const bytes = new Uint8Array(arrayBuffer);
  const img = wasmModule.decode_image(bytes);

  return {
    data: img.get_data(),
    width: img.get_width(),
    height: img.get_height(),
    wbCoeffs: img.get_wb_coeffs(),
    blackLevels: img.get_blacklevels(),
    whiteLevels: img.get_whitelevels(),
    xyzToCam: img.get_xyz_to_cam(),
    orientation: img.get_orientation(),
    make: img.get_make(),
    model: img.get_model(),
    cfaStr: img.get_cfastr(),
    cfaWidth: img.get_cfawidth(),
    crops: img.get_crops(),
  };
}
