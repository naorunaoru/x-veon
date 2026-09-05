import HDR_HISTOGRAM_WGSL from './shaders/histogram-hdr.wgsl?raw';
import HISTOGRAM_REDUCE_WGSL from './shaders/histogram-reduce.wgsl?raw';
import HISTOGRAM_VIZ_WGSL from './shaders/histogram-viz.wgsl?raw';
import { U_FLAGS, U_PREPROCESS } from './uniforms';

export type HistogramMode = 'linear' | 'log' | 'display-linear' | 'display-log';
export type HistogramChannel = 'rgb' | 'luma' | 'ev';

const HIST_BINS = 1024;  // 4 channels × 256 bins
const HIST_BYTES = HIST_BINS * 4;  // 4096 bytes
const DISP_HIST_SIZE = 320;  // downsampled display histogram render target

/** GPU histogram passes and their visualization canvases. */
export class Histogram {
  // Histogram compute resources
  private histogramBinsBuf!: GPUBuffer;
  private hdrHistComputePipeline!: GPUComputePipeline;
  private hdrHistBindGroupLayout!: GPUBindGroupLayout;
  private hdrHistComputeBindGroup: GPUBindGroup | null = null;
  private hdrHistParamsBuf!: GPUBuffer;
  private hdrHistParamsData = new Float32Array(8);  // reused every frame
  private _histogramMode: HistogramMode = 'linear';
  private _histogramChannel: HistogramChannel = 'rgb';

  // Display histogram resources (post-tonemapped)
  private dispHistTex!: GPUTexture;
  private dispHistRenderPipeline!: GPURenderPipeline;
  private dispHistBindGroup!: GPUBindGroup;  // fixed — always reads from dispHistTex

  // Histogram visualization (GPU-rendered)
  private histReducePipeline!: GPUComputePipeline;
  private histReduceBindGroup!: GPUBindGroup;
  private histReduceResultBuf!: GPUBuffer;
  private histReduceFlagBuf!: GPUBuffer;
  private histReduceFlagData = new Uint32Array(1);  // reused for the per-frame flag write
  private histVizPipeline!: GPURenderPipeline;
  private histVizBindGroup!: GPUBindGroup;
  private histVizConfigBuf!: GPUBuffer;
  private histVizConfigData = new Float32Array(4);  // canvas_w, canvas_h, mode, clip_bin
  // Multiple viz targets (e.g. the always-on HUD widget + the open Scopes panel).
  // Each is GPU-rendered to in its own submission so the shared config buffer
  // (per-canvas dimensions) can't race across targets.
  private histVizTargets = new Map<HTMLCanvasElement, GPUCanvasContext>();

  private readonly resources = new Set<GPUBuffer | GPUTexture>();
  constructor(
    private readonly device: GPUDevice,
    shaderModule: GPUShaderModule,
    pipelineLayout: GPUPipelineLayout,
    private readonly uniformBuffer: GPUBuffer,
    private readonly uniformData: Float32Array,
  ) {
    try {
      // ── Histogram resources ───────────────────────────────────────────
      this.histogramBinsBuf = device.createBuffer({
        size: HIST_BYTES,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      this.resources.add(this.histogramBinsBuf);

      // ── HDR histogram compute ───────────────────────────────────────
      this.hdrHistBindGroupLayout = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ],
      });

      const hdrHistComputeModule = device.createShaderModule({ code: HDR_HISTOGRAM_WGSL });
      this.hdrHistComputePipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [this.hdrHistBindGroupLayout] }),
        compute: { module: hdrHistComputeModule, entryPoint: 'main' },
      });

      this.hdrHistParamsBuf = device.createBuffer({
        size: 32,  // 2 × vec4f: (exposure, wb_temp, wb_tint, mode), (range_lo, range_hi, stride, pad)
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.resources.add(this.hdrHistParamsBuf);


      // ── Display histogram resources (post-tonemapped, rendered to float) ──
      this.dispHistTex = device.createTexture({
        size: [DISP_HIST_SIZE, DISP_HIST_SIZE],
        format: 'rgba32float',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
      this.resources.add(this.dispHistTex);

      this.dispHistRenderPipeline = device.createRenderPipeline({
        layout: pipelineLayout,
        vertex: { module: shaderModule, entryPoint: 'vs_main' },
        fragment: {
          module: shaderModule,
          entryPoint: 'fs_main',
          targets: [{ format: 'rgba32float' }],
        },
        primitive: { topology: 'triangle-list' },
      });

      this.dispHistBindGroup = device.createBindGroup({
        layout: this.hdrHistBindGroupLayout,
        entries: [
          { binding: 0, resource: this.dispHistTex.createView() },
          { binding: 1, resource: { buffer: this.histogramBinsBuf } },
          { binding: 2, resource: { buffer: this.hdrHistParamsBuf } },
        ],
      });

      // ── Histogram reduce (scan bins → range + max) ──────────────────
      this.histReduceResultBuf = device.createBuffer({
        size: 16,  // vec4f: (bin_lo, bin_hi, max_log, pad)
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.resources.add(this.histReduceResultBuf);

      this.histReduceFlagBuf = device.createBuffer({
        size: 4,   // u32: force_zero_lo
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.resources.add(this.histReduceFlagBuf);

      const histReduceLayout = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ],
      });

      const histReduceModule = device.createShaderModule({ code: HISTOGRAM_REDUCE_WGSL });
      this.histReducePipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [histReduceLayout] }),
        compute: { module: histReduceModule, entryPoint: 'main' },
      });

      this.histReduceBindGroup = device.createBindGroup({
        layout: histReduceLayout,
        entries: [
          { binding: 0, resource: { buffer: this.histogramBinsBuf } },
          { binding: 1, resource: { buffer: this.histReduceResultBuf } },
          { binding: 2, resource: { buffer: this.histReduceFlagBuf } },
        ],
      });

      // ── Histogram visualization (GPU-rendered) ──────────────────────
      this.histVizConfigBuf = device.createBuffer({
        size: 16,  // vec4f: (canvas_w, canvas_h, mode, clip_bin)
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.resources.add(this.histVizConfigBuf);

      const histVizLayout = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
          { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
          { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        ],
      });

      const histVizModule = device.createShaderModule({ code: HISTOGRAM_VIZ_WGSL });
      this.histVizPipeline = device.createRenderPipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [histVizLayout] }),
        vertex: { module: histVizModule, entryPoint: 'vs_main' },
        fragment: {
          module: histVizModule,
          entryPoint: 'fs_main',
          targets: [{
            format: navigator.gpu.getPreferredCanvasFormat(),
            blend: {
              color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
              alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
            },
          }],
        },
        primitive: { topology: 'triangle-list' },
      });

      this.histVizBindGroup = device.createBindGroup({
        layout: histVizLayout,
        entries: [
          { binding: 0, resource: { buffer: this.histogramBinsBuf } },
          { binding: 1, resource: { buffer: this.histReduceResultBuf } },
          { binding: 2, resource: { buffer: this.histVizConfigBuf } },
        ],
      });

    } catch (error) {
      this.dispose();
      throw error;
    }
  }
  get isDisplayMode(): boolean { return this._histogramMode.startsWith('display-'); }
  get isLogMode(): boolean { return this._histogramMode.endsWith('log'); }
  setMode(mode: HistogramMode): void { this._histogramMode = mode; }
  setChannel(channel: HistogramChannel): void { this._histogramChannel = channel; }
  attach(canvas: HTMLCanvasElement): void {
    if (this.histVizTargets.has(canvas)) return;
    const ctx = canvas.getContext('webgpu');
    if (!ctx) return;
    ctx.configure({
      device: this.device,
      format: navigator.gpu.getPreferredCanvasFormat(),
      alphaMode: 'premultiplied',
    });
    this.histVizTargets.set(canvas, ctx);
  }
  detach(canvas: HTMLCanvasElement): void {
    this.histVizTargets.delete(canvas);
  }
  setImage(imageView: GPUTextureView): void {
    this.hdrHistComputeBindGroup = this.device.createBindGroup({
      layout: this.hdrHistBindGroupLayout,
      entries: [
        { binding: 0, resource: imageView },
        { binding: 1, resource: { buffer: this.histogramBinsBuf } },
        { binding: 2, resource: { buffer: this.hdrHistParamsBuf } },
      ],
    });
  }
  encodeScenePass(encoder: GPUCommandEncoder, imgW: number, imgH: number): void {
    if (!this.hdrHistComputeBindGroup) return;
    // Scene histogram: compute from imageTex in the same submission
    encoder.clearBuffer(this.histogramBinsBuf);

    const p = this.hdrHistParamsData;
    p[0] = this.uniformData[U_PREPROCESS];      // exposure
    p[1] = this.uniformData[U_PREPROCESS + 1];  // wb_temp
    p[2] = this.uniformData[U_PREPROCESS + 2];  // wb_tint
    p[3] = this.isLogMode ? 1.0 : 0.0;                   // mode
    p[4] = this.isLogMode ? -8.0 : 0.0;                  // range_lo
    p[5] = this.isLogMode ? 8.0 : 2.0;                   // range_hi
    p[6] = 4; p[7] = 0;                         // stride, pad
    this.device.queue.writeBuffer(this.hdrHistParamsBuf, 0, p);

    const computePass = encoder.beginComputePass();
    computePass.setPipeline(this.hdrHistComputePipeline);
    computePass.setBindGroup(0, this.hdrHistComputeBindGroup);
    computePass.dispatchWorkgroups(
      Math.ceil(imgW / 4 / 16),
      Math.ceil(imgH / 4 / 16),
    );
    computePass.end();
  }
  renderVizToTargets(): void {
    if (this.histVizTargets.size === 0) return;

    this.histReduceFlagData[0] = this._histogramChannel !== 'ev' ? 1 : 0;
    this.device.queue.writeBuffer(this.histReduceFlagBuf, 0, this.histReduceFlagData);
    const channelCode = this._histogramChannel === 'rgb' ? 0.0 : this._histogramChannel === 'luma' ? 1.0 : 2.0;
    const clipBin = Math.floor((1.0 / 1.2) * 255);

    for (const [canvas, ctx] of this.histVizTargets) {
      // getCurrentTexture() throws if a target's context is no longer usable (canvas
      // detached mid-frame, or device lost). Isolate each target so one bad canvas
      // can't throw out of render() and starve the others — drop it and move on.
      try {
        const c = this.histVizConfigData;
        c[0] = canvas.width;
        c[1] = canvas.height;
        c[2] = channelCode;
        c[3] = clipBin;
        this.device.queue.writeBuffer(this.histVizConfigBuf, 0, c);

        const encoder = this.device.createCommandEncoder();

        // Reduce pass: scan bins → range + max (cheap single workgroup, re-run per canvas).
        const reducePass = encoder.beginComputePass();
        reducePass.setPipeline(this.histReducePipeline);
        reducePass.setBindGroup(0, this.histReduceBindGroup);
        reducePass.dispatchWorkgroups(1);
        reducePass.end();

        // Viz render pass: draw histogram to this canvas.
        const vizView = ctx.getCurrentTexture().createView();
        const vizPass = encoder.beginRenderPass({
          colorAttachments: [{
            view: vizView,
            loadOp: 'clear',
            storeOp: 'store',
            clearValue: { r: 0, g: 0, b: 0, a: 0 },
          }],
        });
        vizPass.setPipeline(this.histVizPipeline);
        vizPass.setBindGroup(0, this.histVizBindGroup);
        vizPass.draw(3);
        vizPass.end();

        this.device.queue.submit([encoder.finish()]);
      } catch {
        this.histVizTargets.delete(canvas);
      }
    }
  }
  renderDisplayHistogram(bindGroup: GPUBindGroup): void {
    const d = this.uniformData;
    d[U_FLAGS + 2] = 1.0;  // exportMode on → raw display-linear output
    // Write only the exportMode float (offset = U_FLAGS+2 floats = (U_FLAGS+2)*4 bytes)
    this.device.queue.writeBuffer(this.uniformBuffer, (U_FLAGS + 2) * 4, d as Float32Array<ArrayBuffer>, U_FLAGS + 2, 1);

    const encoder = this.device.createCommandEncoder();

    // Render tonemapped scene to float texture at reduced resolution
    const histPass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.dispHistTex.createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    histPass.setPipeline(this.dispHistRenderPipeline);
    histPass.setBindGroup(0, bindGroup);
    histPass.draw(3);
    histPass.end();

    // Compute histogram from display texture (exposure/WB zeroed → pass-through)
    encoder.clearBuffer(this.histogramBinsBuf);
    const p = this.hdrHistParamsData;
    p[0] = 0; p[1] = 0; p[2] = 0;       // exposure=0, wb=0 (already baked in)
    p[3] = this.isLogMode ? 1.0 : 0.0;            // mode
    p[4] = this.isLogMode ? -8.0 : 0.0;           // range_lo
    p[5] = this.isLogMode ? 8.0 : 2.0;            // range_hi
    p[6] = 1; p[7] = 0;                  // stride=1 (texture is already small), pad
    this.device.queue.writeBuffer(this.hdrHistParamsBuf, 0, p);

    const computePass = encoder.beginComputePass();
    computePass.setPipeline(this.hdrHistComputePipeline);
    computePass.setBindGroup(0, this.dispHistBindGroup);
    computePass.dispatchWorkgroups(
      Math.ceil(DISP_HIST_SIZE / 16),
      Math.ceil(DISP_HIST_SIZE / 16),
    );
    computePass.end();

    this.device.queue.submit([encoder.finish()]);

    // Display bins are now on the GPU → draw the viz to every registered canvas.
    this.renderVizToTargets();

    // Restore exportMode in CPU-side data; GPU gets it on next render()'s full uniform write
    d[U_FLAGS + 2] = 0.0;
  }
  dispose(): void {
    for (const resource of this.resources) resource.destroy();
    this.resources.clear();
    this.hdrHistComputeBindGroup = null;
    this.histVizTargets.clear();
  }
}
