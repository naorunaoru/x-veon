import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('onnxruntime-web', () => ({ env: { wasm: {}, webgpu: {} }, InferenceSession: { create: vi.fn() } }));

describe('model registry init', () => {
  afterEach(() => { vi.unstubAllGlobals(); vi.resetModules(); });

  it('fails initialisation when the model list cannot be downloaded', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 404 }));
    const { models } = await import('./index');
    await expect(models.init('S')).rejects.toThrow("Couldn't download the model list (HTTP 404).");
  });

  it('fails initialisation when the model list is empty', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) }));
    const { models } = await import('./index');
    await expect(models.init('S')).rejects.toThrow('The model list is empty.');
  });
});
