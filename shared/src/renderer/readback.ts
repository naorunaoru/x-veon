/** GPU→CPU readback helpers shared by the renderer's export and instrumentation paths. */

/** WebGPU requires bytesPerRow to be a multiple of 256. */
export function padTo256(bytes: number): number {
  return Math.ceil(bytes / 256) * 256;
}

/** Read an rgba32float texture back as a tightly packed Float32Array (width × height × 4). */
export async function readTextureRgba(
  device: GPUDevice, texture: GPUTexture, width: number, height: number,
): Promise<Float32Array> {
  const bytesPerRow = padTo256(width * 16);
  const staging = device.createBuffer({
    size: bytesPerRow * height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer(
      { texture },
      { buffer: staging, bytesPerRow, rowsPerImage: height },
      [width, height],
    );
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const src = new Float32Array(staging.getMappedRange());
    const out = new Float32Array(width * height * 4);
    const rowFloats = bytesPerRow / 4;
    for (let y = 0; y < height; y++) {
      out.set(src.subarray(y * rowFloats, y * rowFloats + width * 4), y * width * 4);
    }
    return out;
  } finally {
    if (staging.mapState === 'mapped') staging.unmap();
    staging.destroy();
  }
}

export function f16ToF32(h: number): number {
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

export type ReadbackFormat = 'rgba32float' | 'rgba16float';

/** Off-screen export texture and staging buffer; scene instrumentation stays separate. */
export class ExportTarget {
  private texture: GPUTexture | null = null;
  private staging: GPUBuffer | null = null;
  private width = 0;
  private height = 0;
  private readonly bytesPerPixel: number;
  constructor(private readonly device: GPUDevice, readonly format: ReadbackFormat) {
    this.bytesPerPixel = format === 'rgba32float' ? 16 : 8;
  }
  ensure(width: number, height: number): GPUTextureView {
    if (!this.texture || this.width !== width || this.height !== height) {
      this.dispose();
      try {
        this.texture = this.device.createTexture({
          size: [width, height], format: this.format,
          usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
        });
        this.staging = this.device.createBuffer({
          size: padTo256(width * this.bytesPerPixel) * height,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        this.width = width;
        this.height = height;
      } catch (error) {
        this.dispose();
        throw error;
      }
    }
    return this.texture.createView();
  }
  encodeCopy(encoder: GPUCommandEncoder): void {
    encoder.copyTextureToBuffer(
      { texture: this.texture! },
      { buffer: this.staging!, bytesPerRow: padTo256(this.width * this.bytesPerPixel), rowsPerImage: this.height },
      [this.width, this.height],
    );
  }
  async readHwc(): Promise<Float32Array> {
    const w = this.width, h = this.height;
    const paddedBytesPerRow = padTo256(w * this.bytesPerPixel);
    const staging = this.staging!;
    try {
      await staging.mapAsync(GPUMapMode.READ);
      const mapped = staging.getMappedRange();
      // Convert to HWC RGB Float32Array — no Y-flip needed (WebGPU is top-to-bottom)
      const hwc = new Float32Array(w * h * 3);

      if (this.format === 'rgba32float') {
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

      return hwc;
    } finally {
      if (staging.mapState === 'mapped') staging.unmap();
    }
  }
  dispose(): void {
    this.texture?.destroy();
    this.staging?.destroy();
    this.texture = null;
    this.staging = null;
  }
}
