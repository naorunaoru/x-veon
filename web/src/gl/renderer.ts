// WebGPU HDR preview renderer with OpenDRT tone mapping in a fragment shader.

import type { OpenDrtConfig, TonescaleParams } from './opendrt-params';
import { SRGB_TO_P3D65, P3D65_TO_REC709, P3D65_TO_REC2020, IDENTITY_3X3 } from './color-matrices';
import WGSL_SRC from './shaders/opendrt.wgsl?raw';

export type DisplayGamut = 'rec709' | 'rec2020';

// ── Uniform buffer layout (19 × vec4f = 304 bytes = 76 floats) ──────────

// Float offsets into the uniform buffer shadow array.
const U_TS           = 0;   // vec4: ts_s, ts_s1, ts_m2, ts_dsc
const U_FLAGS        = 4;   // vec4: ts_x0, hdrDisplay, exportMode, ptl_enable
const U_ODRT_TONE    = 8;   // vec4: tn_con, tn_sh, tn_toe, tn_off
const U_ODRT_RS      = 12;  // vec4: rs_sa, rs_rw, rs_bw, 0
const U_ODRT_PT      = 16;  // vec4: pt_r, pt_g, pt_b, pt_rng_low
const U_ODRT_PT2     = 20;  // vec4: pt_rng_high, ptm_high_st, 0, 0
const U_ODRT_LCON    = 24;  // vec4: enable, tn_lcon, tn_lcon_w, tn_lcon_pc
const U_ODRT_HCON    = 28;  // vec4: enable, tn_hcon, tn_hcon_pv, tn_hcon_st
const U_ODRT_BRL_RGB = 32;  // vec4: brl_r, brl_g, brl_b, brl_rng
const U_ODRT_BRL_CMY = 36;  // vec4: brl_c, brl_m, brl_y, enable
const U_ODRT_PTM     = 40;  // vec4: enable, ptm_low, ptm_low_st, ptm_high
const U_PREPROCESS   = 44;  // vec4: exposure, wb_temp, wb_tint, sharpen_amount
const U_TEXEL        = 48;  // vec4: texel_w, texel_h, 0, 0
const U_SRGB_P3_C0   = 52;  // 3 × vec4: columns 0–2 (offsets 52, 56, 60)
const U_P3_DSP_C0    = 64;  // 3 × vec4: columns 0–2 (offsets 64, 68, 72)
const UNIFORM_FLOATS  = 76;
const UNIFORM_BYTES   = UNIFORM_FLOATS * 4; // 304

// ── Module-level device cache ────────────────────────────────────────────

let devicePromise: Promise<GPUDevice> | null = null;

function getDevice(): Promise<GPUDevice> {
  if (!devicePromise) {
    devicePromise = (async () => {
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance',
      });
      if (!adapter) throw new Error('WebGPU adapter not available');

      // Request float32-blendable if supported (needed for rgba32float render target)
      const features: GPUFeatureName[] = [];
      if (adapter.features.has('float32-blendable')) {
        features.push('float32-blendable');
      }

      // Request adapter's max buffer size (default 256 MB is too small for large
      // RGBA32F images — a 6252×4176 photo needs ~398 MB for writeTexture staging).
      const device = await adapter.requestDevice({
        requiredFeatures: features.length > 0 ? features : undefined,
        requiredLimits: {
          maxBufferSize: adapter.limits.maxBufferSize,
        },
      });

      device.lost.then((info) => {
        console.warn('WebGPU device lost:', info.message);
        devicePromise = null;
      });

      return device;
    })();
  }
  return devicePromise;
}

// ── Renderer class ───────────────────────────────────────────────────────

export class HdrRenderer {
  private device: GPUDevice;
  private context: GPUCanvasContext;
  private displayPipeline: GPURenderPipeline;
  private exportPipeline: GPURenderPipeline;
  private bindGroupLayout: GPUBindGroupLayout;
  private sampler: GPUSampler;
  private uniformBuffer: GPUBuffer;
  private uniformData: Float32Array;
  private imageTex: GPUTexture | null = null;
  private bindGroup: GPUBindGroup | null = null;
  private imgW = 0;
  private imgH = 0;
  private _canvas: HTMLCanvasElement;
  private _isHdrDisplay: boolean;
  private _hdrHeadroom: number;
  private _hasFloat32Blendable: boolean;

  // Export resources (lazily created)
  private exportTex: GPUTexture | null = null;
  private exportBuf: GPUBuffer | null = null;
  private exportW = 0;
  private exportH = 0;

  // Saved display state for restore after export render
  private displayTs: TonescaleParams | null = null;
  private displayCfg: OpenDrtConfig | null = null;

  private constructor(
    device: GPUDevice,
    canvas: HTMLCanvasElement,
    context: GPUCanvasContext,
    displayPipeline: GPURenderPipeline,
    exportPipeline: GPURenderPipeline,
    bindGroupLayout: GPUBindGroupLayout,
    sampler: GPUSampler,
    uniformBuffer: GPUBuffer,
    uniformData: Float32Array,
    isHdr: boolean,
    headroom: number,
    hasFloat32Blendable: boolean,
  ) {
    this.device = device;
    this._canvas = canvas;
    this.context = context;
    this.displayPipeline = displayPipeline;
    this.exportPipeline = exportPipeline;
    this.bindGroupLayout = bindGroupLayout;
    this.sampler = sampler;
    this.uniformBuffer = uniformBuffer;
    this.uniformData = uniformData;
    this._isHdrDisplay = isHdr;
    this._hdrHeadroom = headroom;
    this._hasFloat32Blendable = hasFloat32Blendable;
  }

  static async create(
    canvas: HTMLCanvasElement,
    opts?: { hdr?: boolean; headroom?: number },
  ): Promise<HdrRenderer> {
    const device = await getDevice();

    const context = canvas.getContext('webgpu');
    if (!context) throw new Error('WebGPU canvas context not available');

    const wantHdr = opts?.hdr ?? false;
    const headroom = opts?.headroom ?? 1.0;
    const canvasFormat: GPUTextureFormat = wantHdr ? 'rgba16float' : navigator.gpu.getPreferredCanvasFormat();

    context.configure({
      device,
      format: canvasFormat,
      alphaMode: 'opaque',
      colorSpace: wantHdr ? 'display-p3' : 'srgb',
      toneMapping: { mode: wantHdr ? 'extended' : 'standard' },
    });

    const hasFloat32Blendable = device.features.has('float32-blendable');
    const exportFormat: GPUTextureFormat = hasFloat32Blendable ? 'rgba32float' : 'rgba16float';

    // Shader module
    const shaderModule = device.createShaderModule({ code: WGSL_SRC });

    // Bind group layout
    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'non-filtering' } },
      ],
    });

    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    });

    // Display pipeline (targets canvas format)
    const displayPipeline = device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: { module: shaderModule, entryPoint: 'vs_main' },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs_main',
        targets: [{ format: canvasFormat }],
      },
      primitive: { topology: 'triangle-list' },
    });

    // Export pipeline (targets float texture — no blending)
    const exportPipeline = device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: { module: shaderModule, entryPoint: 'vs_main' },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs_main',
        targets: [{ format: exportFormat }],
      },
      primitive: { topology: 'triangle-list' },
    });

    // Sampler (NEAREST, no filtering — matches unfilterable-float)
    const sampler = device.createSampler({
      magFilter: 'nearest',
      minFilter: 'nearest',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    // Uniform buffer
    const uniformBuffer = device.createBuffer({
      size: UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const uniformData = new Float32Array(UNIFORM_FLOATS);

    const renderer = new HdrRenderer(
      device, canvas, context,
      displayPipeline, exportPipeline, bindGroupLayout,
      sampler, uniformBuffer, uniformData,
      wantHdr, headroom, hasFloat32Blendable,
    );

    // Set constant uniforms: matrices + flags
    renderer.setMat3(U_SRGB_P3_C0, SRGB_TO_P3D65);
    renderer.setMat3(U_P3_DSP_C0, wantHdr ? IDENTITY_3X3 : P3D65_TO_REC709);
    uniformData[U_FLAGS + 1] = wantHdr ? 1.0 : 0.0;  // hdrDisplay
    uniformData[U_FLAGS + 2] = 0.0;                    // exportMode

    return renderer;
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

  get imageWidth(): number {
    return this.imgW;
  }

  get imageHeight(): number {
    return this.imgH;
  }

  static isSupported(): boolean {
    return 'gpu' in navigator;
  }

  uploadImage(hwc: Float32Array, width: number, height: number): void {
    if (this.imageTex) this.imageTex.destroy();
    this.imgW = width;
    this.imgH = height;

    // Pad RGB→RGBA (WebGPU has no RGB-only texture formats)
    const pixelCount = width * height;
    const rgba = new Float32Array(pixelCount * 4);
    for (let i = 0; i < pixelCount; i++) {
      rgba[i * 4]     = hwc[i * 3];
      rgba[i * 4 + 1] = hwc[i * 3 + 1];
      rgba[i * 4 + 2] = hwc[i * 3 + 2];
      rgba[i * 4 + 3] = 1.0;
    }

    this.imageTex = this.device.createTexture({
      size: [width, height],
      format: 'rgba32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    this.device.queue.writeTexture(
      { texture: this.imageTex },
      rgba,
      { bytesPerRow: width * 16 },  // 4 channels × 4 bytes
      [width, height],
    );

    // Recreate bind group with new texture
    this.bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: this.imageTex.createView() },
        { binding: 2, resource: this.sampler },
      ],
    });

    // Update texel size
    this.uniformData[U_TEXEL]     = 1 / width;
    this.uniformData[U_TEXEL + 1] = 1 / height;
  }

  setOpenDrtMode(ts: TonescaleParams, cfg: OpenDrtConfig): void {
    this.displayTs = ts;
    this.displayCfg = cfg;
    this.applyOpenDrtUniforms(ts, cfg);
  }

  render(): void {
    if (!this.imageTex || !this.bindGroup) return;

    // Upload uniforms
    this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData as Float32Array<ArrayBuffer>);

    const encoder = this.device.createCommandEncoder();
    const textureView = this.context.getCurrentTexture().createView();

    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: textureView,
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });

    pass.setPipeline(this.displayPipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(3);
    pass.end();

    this.device.queue.submit([encoder.finish()]);
  }

  /**
   * Render OpenDRT to an off-screen texture and read back display-linear pixels.
   *
   * @param cfg OpenDRT config for the export render
   * @param ts  Precomputed tonescale params matching cfg
   * @param gamut Output gamut: 'rec709' for JPEG/TIFF, 'rec2020' for AVIF/HDR
   * @returns Float32Array in HWC layout (width * height * 3), at original (unrotated) dimensions
   */
  async renderForExport(cfg: OpenDrtConfig, ts: TonescaleParams, gamut: DisplayGamut): Promise<Float32Array> {
    if (!this.imageTex || !this.bindGroup) throw new Error('No image uploaded');

    const w = this.imgW;
    const h = this.imgH;

    this.ensureExportResources(w, h);

    // Set export-specific uniforms
    const d = this.uniformData;
    d[U_FLAGS + 1] = 0.0;  // hdrDisplay off (SDR clamp in opendrt())
    d[U_FLAGS + 2] = 1.0;  // exportMode on
    this.setMat3(U_P3_DSP_C0, gamut === 'rec2020' ? P3D65_TO_REC2020 : P3D65_TO_REC709);
    this.applyOpenDrtUniforms(ts, cfg);
    this.device.queue.writeBuffer(this.uniformBuffer, 0, d as Float32Array<ArrayBuffer>);

    // Render to export texture
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.exportTex!.createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    pass.setPipeline(this.exportPipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(3);
    pass.end();

    // Copy texture → readback buffer
    const exportFormat = this._hasFloat32Blendable ? 'rgba32float' : 'rgba16float';
    const bpp = exportFormat === 'rgba32float' ? 16 : 8;
    const paddedBytesPerRow = padTo256(w * bpp);

    encoder.copyTextureToBuffer(
      { texture: this.exportTex! },
      { buffer: this.exportBuf!, bytesPerRow: paddedBytesPerRow, rowsPerImage: h },
      [w, h],
    );

    this.device.queue.submit([encoder.finish()]);

    // Read back (async)
    await this.exportBuf!.mapAsync(GPUMapMode.READ);
    const mapped = this.exportBuf!.getMappedRange();

    // Convert to HWC RGB Float32Array — no Y-flip needed (WebGPU is top-to-bottom)
    const hwc = new Float32Array(w * h * 3);

    if (exportFormat === 'rgba32float') {
      const src = new Float32Array(mapped);
      const paddedRowFloats = paddedBytesPerRow / 4;
      for (let y = 0; y < h; y++) {
        const srcOff = y * paddedRowFloats;
        const dstOff = y * w * 3;
        for (let x = 0; x < w; x++) {
          hwc[dstOff + x * 3]     = src[srcOff + x * 4];
          hwc[dstOff + x * 3 + 1] = src[srcOff + x * 4 + 1];
          hwc[dstOff + x * 3 + 2] = src[srcOff + x * 4 + 2];
        }
      }
    } else {
      // rgba16float fallback: decode float16 → float32
      const src = new Uint16Array(mapped);
      const paddedRowU16 = paddedBytesPerRow / 2;
      for (let y = 0; y < h; y++) {
        const srcOff = y * paddedRowU16;
        const dstOff = y * w * 3;
        for (let x = 0; x < w; x++) {
          hwc[dstOff + x * 3]     = f16ToF32(src[srcOff + x * 4]);
          hwc[dstOff + x * 3 + 1] = f16ToF32(src[srcOff + x * 4 + 1]);
          hwc[dstOff + x * 3 + 2] = f16ToF32(src[srcOff + x * 4 + 2]);
        }
      }
    }

    this.exportBuf!.unmap();

    // Restore display state and re-render
    this.restoreDisplayState();

    return hwc;
  }

  dispose(): void {
    this.imageTex?.destroy();
    this.exportTex?.destroy();
    this.exportBuf?.destroy();
    this.uniformBuffer.destroy();
    // Don't call context.unconfigure() — the canvas context is shared across
    // renderer instances (canvas.getContext('webgpu') returns the same object).
    // In React strict mode, two create() calls race and the cancelled one's
    // dispose would unconfigure the context the surviving renderer needs.
    // The next create() will re-configure, and browser GC handles cleanup.
    this.imageTex = null;
    this.exportTex = null;
    this.exportBuf = null;
    this.bindGroup = null;
  }

  // ── Private helpers ───────────────────────────────────────────────────

  /** Write a row-major 3×3 matrix as 3 column-major vec4s at the given float offset. */
  private setMat3(offset: number, m: Float32Array): void {
    const d = this.uniformData;
    // Column 0: [r0c0, r1c0, r2c0, 0]
    d[offset]     = m[0]; d[offset + 1] = m[3]; d[offset + 2] = m[6]; d[offset + 3] = 0;
    // Column 1: [r0c1, r1c1, r2c1, 0]
    d[offset + 4] = m[1]; d[offset + 5] = m[4]; d[offset + 6] = m[7]; d[offset + 7] = 0;
    // Column 2: [r0c2, r1c2, r2c2, 0]
    d[offset + 8] = m[2]; d[offset + 9] = m[5]; d[offset + 10] = m[8]; d[offset + 11] = 0;
  }

  private applyOpenDrtUniforms(ts: TonescaleParams, cfg: OpenDrtConfig): void {
    const d = this.uniformData;

    // Tonescale params
    d[U_TS]     = ts.ts_s;
    d[U_TS + 1] = ts.ts_s1;
    d[U_TS + 2] = ts.ts_m2;
    d[U_TS + 3] = ts.ts_dsc;
    d[U_FLAGS]  = ts.ts_x0;
    // FLAGS[1] = hdrDisplay — set by caller
    // FLAGS[2] = exportMode — set by caller
    d[U_FLAGS + 3] = cfg.ptl_enable ? 1.0 : 0.0;

    // OpenDRT packed vec4s
    d[U_ODRT_TONE]     = cfg.tn_con;
    d[U_ODRT_TONE + 1] = cfg.tn_sh;
    d[U_ODRT_TONE + 2] = cfg.tn_toe;
    d[U_ODRT_TONE + 3] = cfg.tn_off;

    d[U_ODRT_RS]     = cfg.rs_sa;
    d[U_ODRT_RS + 1] = cfg.rs_rw;
    d[U_ODRT_RS + 2] = cfg.rs_bw;
    d[U_ODRT_RS + 3] = 0;

    d[U_ODRT_PT]     = cfg.pt_r;
    d[U_ODRT_PT + 1] = cfg.pt_g;
    d[U_ODRT_PT + 2] = cfg.pt_b;
    d[U_ODRT_PT + 3] = cfg.pt_rng_low;

    d[U_ODRT_PT2]     = cfg.pt_rng_high;
    d[U_ODRT_PT2 + 1] = cfg.ptm_high_st;
    d[U_ODRT_PT2 + 2] = 0;
    d[U_ODRT_PT2 + 3] = 0;

    d[U_ODRT_LCON]     = cfg.tn_lcon_enable ? 1.0 : 0.0;
    d[U_ODRT_LCON + 1] = cfg.tn_lcon;
    d[U_ODRT_LCON + 2] = cfg.tn_lcon_w;
    d[U_ODRT_LCON + 3] = cfg.tn_lcon_pc;

    d[U_ODRT_HCON]     = cfg.tn_hcon_enable ? 1.0 : 0.0;
    d[U_ODRT_HCON + 1] = cfg.tn_hcon;
    d[U_ODRT_HCON + 2] = cfg.tn_hcon_pv;
    d[U_ODRT_HCON + 3] = cfg.tn_hcon_st;

    d[U_ODRT_BRL_RGB]     = cfg.brl_r;
    d[U_ODRT_BRL_RGB + 1] = cfg.brl_g;
    d[U_ODRT_BRL_RGB + 2] = cfg.brl_b;
    d[U_ODRT_BRL_RGB + 3] = cfg.brl_rng;

    d[U_ODRT_BRL_CMY]     = cfg.brl_c;
    d[U_ODRT_BRL_CMY + 1] = cfg.brl_m;
    d[U_ODRT_BRL_CMY + 2] = cfg.brl_y;
    d[U_ODRT_BRL_CMY + 3] = cfg.brl_enable ? 1.0 : 0.0;

    d[U_ODRT_PTM]     = cfg.ptm_enable ? 1.0 : 0.0;
    d[U_ODRT_PTM + 1] = cfg.ptm_low;
    d[U_ODRT_PTM + 2] = cfg.ptm_low_st;
    d[U_ODRT_PTM + 3] = cfg.ptm_high;

    // Pre-processing
    d[U_PREPROCESS]     = cfg.exposure;
    d[U_PREPROCESS + 1] = cfg.wb_temp;
    d[U_PREPROCESS + 2] = cfg.wb_tint;
    d[U_PREPROCESS + 3] = cfg.sharpen_amount;
  }

  private ensureExportResources(w: number, h: number): void {
    if (this.exportTex && this.exportW === w && this.exportH === h) return;

    this.exportTex?.destroy();
    this.exportBuf?.destroy();

    const exportFormat: GPUTextureFormat = this._hasFloat32Blendable ? 'rgba32float' : 'rgba16float';
    const bpp = exportFormat === 'rgba32float' ? 16 : 8;

    this.exportTex = this.device.createTexture({
      size: [w, h],
      format: exportFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });

    const paddedBytesPerRow = padTo256(w * bpp);
    this.exportBuf = this.device.createBuffer({
      size: paddedBytesPerRow * h,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    this.exportW = w;
    this.exportH = h;
  }

  private restoreDisplayState(): void {
    const d = this.uniformData;

    // Restore flags
    d[U_FLAGS + 1] = this._isHdrDisplay ? 1.0 : 0.0;  // hdrDisplay
    d[U_FLAGS + 2] = 0.0;                               // exportMode off

    // Restore display gamut matrix
    this.setMat3(U_P3_DSP_C0, this._isHdrDisplay ? IDENTITY_3X3 : P3D65_TO_REC709);

    // Restore display OpenDRT config
    if (this.displayTs && this.displayCfg) {
      this.applyOpenDrtUniforms(this.displayTs, this.displayCfg);
    }

    // Re-render to canvas
    this.render();
  }
}

// ── Utilities ────────────────────────────────────────────────────────────

/** WebGPU requires bytesPerRow to be a multiple of 256. */
function padTo256(bytes: number): number {
  return Math.ceil(bytes / 256) * 256;
}

/** Decode a single IEEE 754 half-precision float (uint16) to float32. */
function f16ToF32(h: number): number {
  const s = (h >> 15) & 0x1;
  const e = (h >> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) {
    // Subnormal or zero
    return (s ? -1 : 1) * 2 ** -14 * (m / 1024);
  }
  if (e === 31) {
    return m ? NaN : (s ? -Infinity : Infinity);
  }
  return (s ? -1 : 1) * 2 ** (e - 15) * (1 + m / 1024);
}
