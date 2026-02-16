// WebGL2 HDR preview renderer with OpenDRT tone mapping in a fragment shader.

import type { OpenDrtConfig, TonescaleParams } from './opendrt-params';
import { SRGB_TO_P3D65, P3D65_TO_REC709, IDENTITY_3X3 } from './color-matrices';
import VERT_SRC from './shaders/vert.glsl?raw';
import FRAG_SRC from './shaders/frag.glsl?raw';

// ── Uniform name list ────────────────────────────────────────────────────

const UNIFORM_NAMES = [
  'u_image', 'u_orientation', 'u_toneMapMode', 'u_hdrDisplay', 'u_legacyPeakScale',
  'u_ts_s', 'u_ts_s1', 'u_ts_m2', 'u_ts_dsc', 'u_ts_x0',
  'u_odrt_tone', 'u_odrt_rs', 'u_odrt_pt', 'u_odrt_pt2',
  'u_odrt_lcon', 'u_odrt_hcon',
  'u_odrt_brl_rgb', 'u_odrt_brl_cmy',
  'u_odrt_ptm', 'u_odrt_ptm_high_st', 'u_odrt_ptl_enable',
  'u_srgbToP3', 'u_p3ToDisplay',
] as const;

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
  private peakScale = 1.0;
  private _isHdrDisplay = false;
  private _hdrHeadroom = 1.0;

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

    // Compute peak for legacy mode
    let peak = 0;
    for (let i = 0; i < hwc.length; i++) {
      if (hwc[i] > peak) peak = hwc[i];
    }
    this.peakScale = peak > 1.0 ? 1.0 / peak : 1.0;
  }

  setOrientation(orient: number): void {
    const gl = this.gl;
    gl.useProgram(this.program);
    gl.uniform1i(this.loc('u_orientation'), orient);
  }

  setLegacyMode(): void {
    const gl = this.gl;
    gl.useProgram(this.program);
    gl.uniform1i(this.loc('u_toneMapMode'), 0);
    // HDR: don't compress super-whites — let the display clip at its native peak
    gl.uniform1f(this.loc('u_legacyPeakScale'), this._isHdrDisplay ? 1.0 : this.peakScale);
  }

  setOpenDrtMode(ts: TonescaleParams, cfg: OpenDrtConfig): void {

    const gl = this.gl;
    gl.useProgram(this.program);
    gl.uniform1i(this.loc('u_toneMapMode'), 1);

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

  dispose(): void {
    const gl = this.gl;
    if (this.tex) gl.deleteTexture(this.tex);
    gl.deleteVertexArray(this.vao);
    gl.deleteProgram(this.program);
    this.tex = null;
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
