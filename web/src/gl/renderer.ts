// WebGL2 HDR preview renderer with OpenDRT tone mapping in a fragment shader.

import type { OpenDrtConfig, TonescaleParams } from './opendrt-params';
import { SRGB_TO_P3D65, P3D65_TO_REC709, P3D65_TO_REC2020, IDENTITY_3X3 } from './color-matrices';
import VERT_SRC from './shaders/vert.glsl?raw';
import FRAG_SRC from './shaders/frag.glsl?raw';

// ── Uniform name list ────────────────────────────────────────────────────

const UNIFORM_NAMES = [
  'u_image', 'u_hdrDisplay', 'u_exportMode',
  'u_ts_s', 'u_ts_s1', 'u_ts_m2', 'u_ts_dsc', 'u_ts_x0',
  'u_odrt_tone', 'u_odrt_rs', 'u_odrt_pt', 'u_odrt_pt2',
  'u_odrt_lcon', 'u_odrt_hcon',
  'u_odrt_brl_rgb', 'u_odrt_brl_cmy',
  'u_odrt_ptm', 'u_odrt_ptm_high_st', 'u_odrt_ptl_enable',
  'u_srgbToP3', 'u_p3ToDisplay',
  'u_exposure', 'u_wb_temp', 'u_wb_tint',
] as const;

export type DisplayGamut = 'rec709' | 'rec2020';

// ── Renderer class ───────────────────────────────────────────────────────

export class HdrRenderer {
  private gl: WebGL2RenderingContext;
  private program: WebGLProgram;
  private tex: WebGLTexture | null = null;
  private vao: WebGLVertexArrayObject;
  private locs: Map<string, WebGLUniformLocation | null>;
  private imgW = 0;
  private imgH = 0;
  private _canvas: HTMLCanvasElement;
  private _isHdrDisplay = false;
  private _hdrHeadroom = 1.0;

  // Export FBO (lazily created)
  private exportFbo: WebGLFramebuffer | null = null;
  private exportTex: WebGLTexture | null = null;
  private exportW = 0;
  private exportH = 0;

  // Saved display state for restore after export render
  private displayTs: TonescaleParams | null = null;
  private displayCfg: OpenDrtConfig | null = null;

  constructor(canvas: HTMLCanvasElement, opts?: { hdr?: boolean; headroom?: number }) {
    this._canvas = canvas;
    const gl = canvas.getContext('webgl2', {
      alpha: false,
      depth: false,
      stencil: false,
      antialias: false,
      premultipliedAlpha: false,
      preserveDrawingBuffer: true,
    });
    if (!gl) throw new Error('WebGL2 not available');
    this.gl = gl;

    // Enable float color buffer attachment (required for RGBA32F FBO)
    gl.getExtension('EXT_color_buffer_float');

    // Configure HDR display output (float16 backbuffer + P3 + extended range)
    if (opts?.hdr) {
      try {
        if (typeof gl.drawingBufferStorage === 'function') {
          gl.drawingBufferStorage(gl.RGBA16F, canvas.width, canvas.height);
        }
        if ('drawingBufferColorSpace' in gl) {
          gl.drawingBufferColorSpace = 'display-p3';
        }
        // Extended tone mapping: try new API first, then legacy
        let extOk = false;
        if (typeof gl.drawingBufferToneMapping === 'function') {
          try { gl.drawingBufferToneMapping({ mode: 'extended' }); extOk = true; } catch { /* */ }
        }
        if (!extOk && typeof canvas.configureHighDynamicRange === 'function') {
          try { canvas.configureHighDynamicRange({ mode: 'extended' }); extOk = true; } catch { /* */ }
        }
        if (extOk) {
          this._isHdrDisplay = true;
          this._hdrHeadroom = opts.headroom ?? 2.0;
        }
      } catch {
        // Fall back to SDR silently
      }
    }

    // Compile and link
    const vs = this.compileShader(gl.VERTEX_SHADER, VERT_SRC);
    const fs = this.compileShader(gl.FRAGMENT_SHADER, FRAG_SRC);
    this.program = gl.createProgram()!;
    gl.attachShader(this.program, vs);
    gl.attachShader(this.program, fs);
    gl.linkProgram(this.program);
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
      throw new Error('Shader link failed: ' + gl.getProgramInfoLog(this.program));
    }
    gl.deleteShader(vs);
    gl.deleteShader(fs);

    // Cache uniform locations
    this.locs = new Map();
    for (const name of UNIFORM_NAMES) {
      this.locs.set(name, gl.getUniformLocation(this.program, name));
    }

    // Empty VAO for attributeless rendering
    this.vao = gl.createVertexArray()!;

    // Set constant uniforms
    gl.useProgram(this.program);
    this.setMat3('u_srgbToP3', SRGB_TO_P3D65);
    this.setMat3('u_p3ToDisplay', this._isHdrDisplay ? IDENTITY_3X3 : P3D65_TO_REC709);
    gl.uniform1i(this.loc('u_hdrDisplay'), this._isHdrDisplay ? 1 : 0);
    gl.uniform1i(this.loc('u_exportMode'), 0);
    gl.uniform1i(this.loc('u_image'), 0);
  }

  get canvas(): HTMLCanvasElement {
    return this._canvas;
  }

  get isHdrDisplay(): boolean {
    return this._isHdrDisplay;
  }

  get hdrHeadroom(): number {
    return this._hdrHeadroom;
  }

  /** Original (unrotated) image dimensions. */
  get imageWidth(): number {
    return this.imgW;
  }

  get imageHeight(): number {
    return this.imgH;
  }

  static isSupported(): boolean {
    try {
      const c = document.createElement('canvas');
      const gl = c.getContext('webgl2');
      if (!gl) return false;
      // WebGL2 natively supports float textures for sampling with NEAREST.
      // EXT_color_buffer_float is only needed for rendering TO float textures.
      gl.getExtension('WEBGL_lose_context')?.loseContext();
      return true;
    } catch {
      return false;
    }
  }

  uploadImage(hwc: Float32Array, width: number, height: number): void {
    const gl = this.gl;
    if (this.tex) gl.deleteTexture(this.tex);
    this.tex = gl.createTexture()!;
    this.imgW = width;
    this.imgH = height;

    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    // Try RGB32F first (matches HWC layout), fall back to RGBA32F
    const err = gl.getError(); // clear any pending
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGB32F,
      width, height, 0,
      gl.RGB, gl.FLOAT, hwc,
    );
    if (gl.getError() !== gl.NO_ERROR) {
      // Fallback: pad to RGBA
      const rgba = new Float32Array(width * height * 4);
      for (let i = 0; i < width * height; i++) {
        rgba[i * 4] = hwc[i * 3];
        rgba[i * 4 + 1] = hwc[i * 3 + 1];
        rgba[i * 4 + 2] = hwc[i * 3 + 2];
        rgba[i * 4 + 3] = 1.0;
      }
      gl.texImage2D(
        gl.TEXTURE_2D, 0, gl.RGBA32F,
        width, height, 0,
        gl.RGBA, gl.FLOAT, rgba,
      );
    }

  }

  setOpenDrtMode(ts: TonescaleParams, cfg: OpenDrtConfig): void {
    this.displayTs = ts;
    this.displayCfg = cfg;
    this.applyOpenDrtUniforms(ts, cfg);
  }

  render(): void {
    const gl = this.gl;
    if (!this.tex) return;

    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  /**
   * Render OpenDRT to an off-screen FBO and read back display-linear pixels.
   *
   * @param cfg OpenDRT config for the export render
   * @param ts  Precomputed tonescale params matching cfg
   * @param gamut Output gamut: 'rec709' for JPEG/TIFF, 'rec2020' for AVIF/HDR
   * @returns Float32Array in HWC layout (width * height * 3), at original (unrotated) dimensions
   */
  renderForExport(cfg: OpenDrtConfig, ts: TonescaleParams, gamut: DisplayGamut): Float32Array {
    const gl = this.gl;
    if (!this.tex) throw new Error('No image uploaded');

    const w = this.imgW;
    const h = this.imgH;

    // Ensure FBO exists at the right dimensions
    this.ensureExportFbo(w, h);

    gl.useProgram(this.program);

    // Set export-specific uniforms
    gl.uniform1i(this.loc('u_exportMode'), 1);
    gl.uniform1i(this.loc('u_hdrDisplay'), 0);  // SDR clamp path in opendrt()
    this.setMat3('u_p3ToDisplay', gamut === 'rec2020' ? P3D65_TO_REC2020 : P3D65_TO_REC709);
    this.applyOpenDrtUniforms(ts, cfg);

    // Render to FBO
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.exportFbo);
    gl.viewport(0, 0, w, h);
    gl.bindVertexArray(this.vao);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // Read back RGBA float pixels
    const rgba = new Float32Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, rgba);

    // Unbind FBO
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    // Restore display state
    this.restoreDisplayState();

    // Strip alpha → HWC RGB, flip Y (GL readPixels is bottom-up)
    const hwc = new Float32Array(w * h * 3);
    for (let y = 0; y < h; y++) {
      const srcRow = (h - 1 - y) * w;
      const dstRow = y * w;
      for (let x = 0; x < w; x++) {
        const si = (srcRow + x) * 4;
        const di = (dstRow + x) * 3;
        hwc[di]     = rgba[si];
        hwc[di + 1] = rgba[si + 1];
        hwc[di + 2] = rgba[si + 2];
      }
    }

    return hwc;
  }

  dispose(): void {
    const gl = this.gl;
    if (this.tex) gl.deleteTexture(this.tex);
    if (this.exportFbo) gl.deleteFramebuffer(this.exportFbo);
    if (this.exportTex) gl.deleteTexture(this.exportTex);
    gl.deleteVertexArray(this.vao);
    gl.deleteProgram(this.program);
    this.tex = null;
    this.exportFbo = null;
    this.exportTex = null;
  }

  // ── Private helpers ───────────────────────────────────────────────────

  private loc(name: string): WebGLUniformLocation | null {
    return this.locs.get(name) ?? null;
  }

  private setMat3(name: string, m: Float32Array): void {
    // gl.uniformMatrix3fv expects column-major order.
    // Our matrices are row-major, so transpose.
    const col = new Float32Array([
      m[0], m[3], m[6],
      m[1], m[4], m[7],
      m[2], m[5], m[8],
    ]);
    this.gl.uniformMatrix3fv(this.loc(name), false, col);
  }

  private applyOpenDrtUniforms(ts: TonescaleParams, cfg: OpenDrtConfig): void {
    const gl = this.gl;
    gl.useProgram(this.program);

    // Tonescale params
    gl.uniform1f(this.loc('u_ts_s'), ts.ts_s);
    gl.uniform1f(this.loc('u_ts_s1'), ts.ts_s1);
    gl.uniform1f(this.loc('u_ts_m2'), ts.ts_m2);
    gl.uniform1f(this.loc('u_ts_dsc'), ts.ts_dsc);
    gl.uniform1f(this.loc('u_ts_x0'), ts.ts_x0);

    // Config packed into vec4s
    gl.uniform4f(this.loc('u_odrt_tone'), cfg.tn_con, cfg.tn_sh, cfg.tn_toe, cfg.tn_off);
    gl.uniform4f(this.loc('u_odrt_rs'), cfg.rs_sa, cfg.rs_rw, cfg.rs_bw, 0);
    gl.uniform4f(this.loc('u_odrt_pt'), cfg.pt_r, cfg.pt_g, cfg.pt_b, cfg.pt_rng_low);
    gl.uniform4f(this.loc('u_odrt_pt2'), cfg.pt_rng_high, 0, 0, 0);
    gl.uniform4f(this.loc('u_odrt_lcon'),
      cfg.tn_lcon_enable ? 1.0 : 0.0, cfg.tn_lcon, cfg.tn_lcon_w, cfg.tn_lcon_pc);
    gl.uniform4f(this.loc('u_odrt_hcon'),
      cfg.tn_hcon_enable ? 1.0 : 0.0, cfg.tn_hcon, cfg.tn_hcon_pv, cfg.tn_hcon_st);
    gl.uniform4f(this.loc('u_odrt_brl_rgb'), cfg.brl_r, cfg.brl_g, cfg.brl_b, cfg.brl_rng);
    gl.uniform4f(this.loc('u_odrt_brl_cmy'),
      cfg.brl_c, cfg.brl_m, cfg.brl_y, cfg.brl_enable ? 1.0 : 0.0);
    gl.uniform4f(this.loc('u_odrt_ptm'),
      cfg.ptm_enable ? 1.0 : 0.0, cfg.ptm_low, cfg.ptm_low_st, cfg.ptm_high);
    gl.uniform1f(this.loc('u_odrt_ptm_high_st'), cfg.ptm_high_st);
    gl.uniform1f(this.loc('u_odrt_ptl_enable'), cfg.ptl_enable ? 1.0 : 0.0);

    // Pre-processing uniforms
    gl.uniform1f(this.loc('u_exposure'), cfg.exposure);
    gl.uniform1f(this.loc('u_wb_temp'), cfg.wb_temp);
    gl.uniform1f(this.loc('u_wb_tint'), cfg.wb_tint);
  }

  private ensureExportFbo(w: number, h: number): void {
    const gl = this.gl;

    if (this.exportFbo && this.exportW === w && this.exportH === h) return;

    // Clean up old
    if (this.exportFbo) gl.deleteFramebuffer(this.exportFbo);
    if (this.exportTex) gl.deleteTexture(this.exportTex);

    // Create RGBA32F texture for FBO attachment
    this.exportTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.exportTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, w, h, 0, gl.RGBA, gl.FLOAT, null);

    // Create FBO
    this.exportFbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.exportFbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.exportTex, 0);

    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error(`Export FBO incomplete: 0x${status.toString(16)}`);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.exportW = w;
    this.exportH = h;
  }

  private restoreDisplayState(): void {
    const gl = this.gl;
    gl.useProgram(this.program);

    // Restore export mode off
    gl.uniform1i(this.loc('u_exportMode'), 0);

    // Restore display gamut
    this.setMat3('u_p3ToDisplay', this._isHdrDisplay ? IDENTITY_3X3 : P3D65_TO_REC709);
    gl.uniform1i(this.loc('u_hdrDisplay'), this._isHdrDisplay ? 1 : 0);

    // Restore display OpenDRT config
    if (this.displayTs && this.displayCfg) {
      this.applyOpenDrtUniforms(this.displayTs, this.displayCfg);
    }

    // Re-render to canvas
    this.render();
  }

  private compileShader(type: number, src: string): WebGLShader {
    const gl = this.gl;
    const s = gl.createShader(type)!;
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(s);
      gl.deleteShader(s);
      throw new Error('Shader compile failed: ' + info);
    }
    return s;
  }
}
