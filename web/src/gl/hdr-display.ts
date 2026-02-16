// Probe WebGL2 HDR display capabilities (float16 backbuffer + extended range).

export interface HdrDisplayInfo {
  /** Whether the browser supports float16 backbuffer + extended range in WebGL2. */
  supported: boolean;
  /** Peak-to-SDR luminance ratio (e.g. 4.0 = 400 nit peak). 1.0 if unknown/SDR. */
  headroom: number;
}

/** Read peak/SDR luminance ratio from screen API or media query. */
export function getHdrHeadroom(): number {
  if (typeof screen !== 'undefined' && screen.highDynamicRangeHeadroom != null) {
    const hr = screen.highDynamicRangeHeadroom;
    if (typeof hr === 'number' && hr > 1.0) return hr;
  }
  if (typeof matchMedia !== 'undefined' && matchMedia('(dynamic-range: high)').matches) {
    return 2.0; // conservative default
  }
  return 1.0;
}

/**
 * Probe WebGL2 canvas HDR capability using a throwaway context.
 * Tries drawingBufferStorage(RGBA16F) + display-p3 + extended tone mapping.
 */
export function probeHdrDisplay(): HdrDisplayInfo {
  const fail: HdrDisplayInfo = { supported: false, headroom: 1.0 };
  try {
    const canvas = document.createElement('canvas');
    canvas.width = 1;
    canvas.height = 1;

    const gl = canvas.getContext('webgl2', {
      alpha: false,
      depth: false,
      stencil: false,
      antialias: false,
      premultipliedAlpha: false,
    });
    if (!gl) return fail;

    const cleanup = () => gl.getExtension('WEBGL_lose_context')?.loseContext();

    // 1. Float16 backbuffer
    if (typeof gl.drawingBufferStorage !== 'function') { cleanup(); return fail; }
    gl.drawingBufferStorage(gl.RGBA16F, 1, 1);

    // 2. Display P3 color space
    try {
      gl.drawingBufferColorSpace = 'display-p3';
    } catch {
      cleanup();
      return fail;
    }

    // 3. Extended tone mapping (new API first, then legacy)
    let extendedOk = false;
    if (typeof gl.drawingBufferToneMapping === 'function') {
      try { gl.drawingBufferToneMapping({ mode: 'extended' }); extendedOk = true; } catch { /* */ }
    }
    if (!extendedOk && typeof canvas.configureHighDynamicRange === 'function') {
      try { canvas.configureHighDynamicRange({ mode: 'extended' }); extendedOk = true; } catch { /* */ }
    }

    cleanup();
    if (!extendedOk) return fail;

    return { supported: true, headroom: getHdrHeadroom() };
  } catch {
    return fail;
  }
}
