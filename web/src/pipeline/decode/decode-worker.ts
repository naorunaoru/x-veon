/// <reference lib="webworker" />
/**
 * RAW decoding, and the CPU preparation of the CFA (crop, white-level calibration, phase
 * padding, normalisation table), off the main thread. Only the prepared u16 CFA comes back, as a
 * transfer; the full readout stays here and is dropped. rawloader's panics trap the wasm instance (catch_unwind does
 * nothing on wasm32) and leave its stack and heap corrupted, so the owner replaces this worker
 * after any failed decode instead of reusing the instance.
 */
import init, { decode_image } from '../../../../shared/crates/rawloader-wasm/pkg/rawloader_wasm.js';
import { prepareCfa } from '../preprocess/preprocessor';
import type { PreparedCfa, RawImage } from '../types';

const ready = init();

type Request = { type: 'ping' } | { type: 'decode'; bytes: ArrayBuffer };

self.onmessage = async (e: MessageEvent<Request>) => {
  try {
    await ready;
    if (e.data.type === 'ping') {
      // Readiness probe: lets initWasm fail early if the module can't load.
      self.postMessage({ type: 'pong' });
      return;
    }
    const img = decode_image(new Uint8Array(e.data.bytes));
    let raw: RawImage;
    try {
      raw = {
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
        drGain: img.get_dr_gain(),
        camToXyz: img.get_cam_to_xyz(),
        exposureBias: img.get_exposure_bias(),
        lensModel: img.get_lens_model(),
        focalLength: img.get_focal_length(),
        fNumber: img.get_f_number(),
      };
    } finally {
      img.free();
    }
    // A readout the pipeline can't lay out (an unsupported CFA) decoded fine: report that
    // separately, so it isn't mistaken for a decoder failure and the instance stays usable.
    let prepared: PreparedCfa | null = null;
    let prepareError: string | null = null;
    try {
      prepared = prepareCfa(raw);
    } catch (err) {
      prepareError = err instanceof Error ? err.message : String(err);
    }
    const { data: _data, ...meta } = raw;
    const transfer: Transferable[] = prepared ? [prepared.data.buffer, prepared.lut.buffer] : [];
    self.postMessage({ type: 'done', raw: meta, prepared, prepareError }, transfer);
  } catch (err) {
    self.postMessage({ type: 'error', message: err instanceof Error ? err.message : String(err) });
  }
};

