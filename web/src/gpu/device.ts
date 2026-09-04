/**
 * Shared WebGPU device provider. ONNX Runtime creates the device during model init and
 * registers it here; everyone else takes it from getDevice(). If nothing registered a
 * device, one is created on first use.
 */
let devicePromise: Promise<GPUDevice> | null = null;

export function setSharedDevice(device: GPUDevice): void {
  devicePromise = Promise.resolve(device);
}

export function getDevice(): Promise<GPUDevice> {
  if (!devicePromise) {
    devicePromise = (async () => {
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance',
      });
      if (!adapter) throw new Error('WebGPU adapter not available');

      const features: GPUFeatureName[] = [];
      if (adapter.features.has('float32-blendable')) features.push('float32-blendable');

      const device = await adapter.requestDevice({
        requiredFeatures: features.length > 0 ? features : undefined,
        requiredLimits: {
          maxBufferSize: adapter.limits.maxBufferSize,
          maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
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
