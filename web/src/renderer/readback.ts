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
