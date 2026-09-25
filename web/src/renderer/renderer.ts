// WebGPU display facade; histogram and readback resources have separate owners.
import type { GradingConfig, TonescaleParams } from './grading/opendrt-params';
import type { GpuImage } from '@/lib/types';
import { SRGB_TO_P3D65, P3D65_TO_REC709, IDENTITY_3X3 } from './color-matrices';
import WGSL_SRC from './shaders/opendrt.wgsl?raw';
import { Histogram, type HistogramMode, type HistogramChannel } from './histogram';
import { ExportTarget, readTextureRgba } from './readback';
import { U_FLAGS, U_TEXEL, U_SRGB_P3_C0, U_P3_DSP_C0, UNIFORM_BYTES, UNIFORM_FLOATS, setMat3, applyOpenDrtUniforms } from './uniforms';
import { getDevice } from '@/gpu/device';

/** Full float precision for exports; half floats would leave a "16-bit" TIFF with ~11 bits. */
const EXPORT_FORMAT = 'rgba32float' as const;
export type DisplayGamut = 'rec709' | 'rec2020';
export interface HistogramControls {
  attach(canvas: HTMLCanvasElement): void;
  detach(canvas: HTMLCanvasElement): void;
  setMode(mode: HistogramMode): void;
  setChannel(channel: HistogramChannel): void;
}
export interface Renderer {
  readonly display: { hdr: boolean; headroom: number };
  /** Switch between SDR and HDR output (or a new headroom) in place: same image, no re-upload. */
  setDisplay(display: { hdr: boolean; headroom: number }): void;
  /** Samples the image's texture directly; the caller keeps ownership and keeps it alive. */
  setImage(image: GpuImage): void;
  setGrade(cfg: GradingConfig, ts: TonescaleParams): void;
  render(): void;
  readonly histogram: HistogramControls;
  readback(cfg: GradingConfig, ts: TonescaleParams, gamut: DisplayGamut): Promise<Float32Array>;
  readbackImage(): Promise<Float32Array>;
  dispose(): void;
}

export class HdrRenderer implements Renderer {
  private device: GPUDevice;
  private context: GPUCanvasContext;
  private displayPipeline: GPURenderPipeline;
  private exportPipeline: GPURenderPipeline;
  private bindGroupLayout: GPUBindGroupLayout;
  private sampler: GPUSampler;
  private shaderModule: GPUShaderModule;
  private pipelineLayout: GPUPipelineLayout;
  private canvasFormat: GPUTextureFormat;
  private uniformBuffer: GPUBuffer;
  private uniformData: Float32Array;
  private imageTex: GPUTexture | null = null;
  private bindGroup: GPUBindGroup | null = null;
  private imgW = 0;
  private imgH = 0;
  private _isHdrDisplay: boolean;
  private _hdrHeadroom: number;

  readonly histogram: Histogram;
  private readonly exportTarget: ExportTarget;
  private displayTs: TonescaleParams | null = null;
  private displayCfg: GradingConfig | null = null;
  private constructor(
    device: GPUDevice,
    context: GPUCanvasContext,
    displayPipeline: GPURenderPipeline,
    exportPipeline: GPURenderPipeline,
    bindGroupLayout: GPUBindGroupLayout,
    sampler: GPUSampler,
    uniformBuffer: GPUBuffer,
    uniformData: Float32Array,
    isHdr: boolean,
    headroom: number,
    shaderModule: GPUShaderModule,
    pipelineLayout: GPUPipelineLayout,
  ) {
    this.device = device;
    this.context = context;
    this.displayPipeline = displayPipeline;
    this.exportPipeline = exportPipeline;
    this.bindGroupLayout = bindGroupLayout;
    this.sampler = sampler;
    this.shaderModule = shaderModule;
    this.pipelineLayout = pipelineLayout;
    this.canvasFormat = displayFormat(isHdr);
    this.uniformBuffer = uniformBuffer;
    this.uniformData = uniformData;
    this._isHdrDisplay = isHdr;
    this._hdrHeadroom = headroom;
    this.histogram = new Histogram(device, shaderModule, pipelineLayout, uniformBuffer, uniformData);
    this.exportTarget = new ExportTarget(device, EXPORT_FORMAT);
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
    const canvasFormat = displayFormat(wantHdr);
    configureContext(context, device, wantHdr);


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
    const displayPipeline = createDisplayPipeline(device, pipelineLayout, shaderModule, canvasFormat);

    // Export pipeline: renders to rgba32float. It never blends, so it doesn't need the
    // float32-blendable feature (which ONNX Runtime's shared device doesn't request).
    const exportPipeline = device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: { module: shaderModule, entryPoint: 'vs_main' },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs_main',
        targets: [{ format: EXPORT_FORMAT }],
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

    const uniformData = new Float32Array(UNIFORM_FLOATS);

    // Uniform buffer
    const uniformBuffer = device.createBuffer({
      size: UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    try {
      const renderer = new HdrRenderer(
        device, context, displayPipeline, exportPipeline, bindGroupLayout,
        sampler, uniformBuffer, uniformData, wantHdr, headroom,
        shaderModule, pipelineLayout,
      );
      // Set constant uniforms: matrices + flags
      setMat3(uniformData, U_SRGB_P3_C0, SRGB_TO_P3D65);
      uniformData[U_FLAGS + 2] = 0.0;                    // exportMode
      renderer.restoreDisplayState(false);

      return renderer;
    } catch (error) {
      uniformBuffer.destroy();
      throw error;
    }
  }
  get display(): { hdr: boolean; headroom: number } {
    return { hdr: this._isHdrDisplay, headroom: this._hdrHeadroom };
  }
  setDisplay(display: { hdr: boolean; headroom: number }): void {
    this._hdrHeadroom = display.headroom;
    if (display.hdr !== this._isHdrDisplay) {
      this._isHdrDisplay = display.hdr;
      const format = displayFormat(display.hdr);
      configureContext(this.context, this.device, display.hdr);
      if (format !== this.canvasFormat) {
        this.canvasFormat = format;
        this.displayPipeline = createDisplayPipeline(this.device, this.pipelineLayout, this.shaderModule, format);
      }
    }
    // The caller sets the grade for the new headroom; this restores flags and gamut and draws.
    this.restoreDisplayState(false);
  }

  setImage(image: GpuImage): void {
    this.imageTex = image.texture;
    this.imgW = image.width;
    this.imgH = image.height;
    this.uniformData[U_TEXEL]     = 1 / image.width;
    this.uniformData[U_TEXEL + 1] = 1 / image.height;
    this.rebuildBindGroups();
  }

  private rebuildBindGroups(): void {
    this.bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: this.imageTex!.createView() },
        { binding: 2, resource: this.sampler },
      ],
    });

    this.histogram.setImage(this.imageTex!.createView());
  }

  setGrade(cfg: GradingConfig, ts: TonescaleParams): void {
    this.displayTs = ts;
    this.displayCfg = cfg;
    applyOpenDrtUniforms(this.uniformData, ts, cfg, this._isHdrDisplay ? 'p3' : 'rec709');
  }

  render(): void {
    if (!this.imageTex || !this.bindGroup) return;

    // Upload uniforms
    this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData as Float32Array<ArrayBuffer>);

    const encoder = this.device.createCommandEncoder();

    // ── Display render pass ───────────────────────────────────────────
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

    const isDisplay = this.histogram.isDisplayMode;
    if (!isDisplay) this.histogram.encodeScenePass(encoder, this.imgW, this.imgH);
    this.device.queue.submit([encoder.finish()]);
    if (!isDisplay) this.histogram.renderVizToTargets();
    else this.histogram.renderDisplayHistogram(this.bindGroup);
  }

  async readback(cfg: GradingConfig, ts: TonescaleParams, gamut: DisplayGamut): Promise<Float32Array> {
    if (!this.imageTex || !this.bindGroup) throw new Error('No image uploaded');

    const view = this.exportTarget.ensure(this.imgW, this.imgH);
    try {

      // Set export-specific uniforms
      const d = this.uniformData;
      d[U_FLAGS + 1] = 0.0;  // hdrDisplay off (SDR clamp in opendrt())
      d[U_FLAGS + 2] = 1.0;  // exportMode on
      applyOpenDrtUniforms(this.uniformData, ts, cfg, gamut);
      this.device.queue.writeBuffer(this.uniformBuffer, 0, d as Float32Array<ArrayBuffer>);

      // Render to export texture
      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view,
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        }],
      });
      pass.setPipeline(this.exportPipeline);
      pass.setBindGroup(0, this.bindGroup);
      pass.draw(3);
      pass.end();

      this.exportTarget.encodeCopy(encoder);
      this.device.queue.submit([encoder.finish()]);
      return await this.exportTarget.readHwc();
    } finally {
      this.restoreDisplayState();
    }
  }

  async readbackImage(): Promise<Float32Array> {
    if (!this.imageTex) throw new Error('No image uploaded');
    return readTextureRgba(this.device, this.imageTex, this.imgW, this.imgH);
  }

  dispose(): void {
    // The image texture belongs to the processed result, not to the renderer.
    this.uniformBuffer.destroy();
    this.histogram.dispose();
    this.exportTarget.dispose();
    // The canvas context is shared across strict-mode renderer instances.
    this.imageTex = null;
    this.bindGroup = null;
  }

  /** Put display flags, gamut and grade back after an export (or a display change), then draw. */
  private restoreDisplayState(draw = true): void {
    const d = this.uniformData;

    // Restore flags
    d[U_FLAGS + 1] = this._isHdrDisplay ? 1.0 : 0.0;  // hdrDisplay
    d[U_FLAGS + 2] = 0.0;                               // exportMode off

    // Restore display gamut matrix
    setMat3(this.uniformData, U_P3_DSP_C0, this._isHdrDisplay ? IDENTITY_3X3 : P3D65_TO_REC709);

    // Restore display OpenDRT config
    if (this.displayTs && this.displayCfg) {
      applyOpenDrtUniforms(this.uniformData, this.displayTs, this.displayCfg, this._isHdrDisplay ? 'p3' : 'rec709');
    }

    // Re-render to canvas
    if (draw) this.render();
  }
}

function displayFormat(hdr: boolean): GPUTextureFormat {
  return hdr ? 'rgba16float' : navigator.gpu.getPreferredCanvasFormat();
}

function configureContext(context: GPUCanvasContext, device: GPUDevice, hdr: boolean): void {
  context.configure({
    device,
    format: displayFormat(hdr),
    alphaMode: 'opaque',
    colorSpace: hdr ? 'display-p3' : 'srgb',
    toneMapping: { mode: hdr ? 'extended' : 'standard' },
  });
}

function createDisplayPipeline(
  device: GPUDevice, layout: GPUPipelineLayout, module: GPUShaderModule, format: GPUTextureFormat,
): GPURenderPipeline {
  return device.createRenderPipeline({
    layout,
    vertex: { module, entryPoint: 'vs_main' },
    fragment: { module, entryPoint: 'fs_main', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });
}
