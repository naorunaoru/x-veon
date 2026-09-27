import { describe, expect, it } from 'vitest';
import { getDevice, setSharedDevice } from './device';

describe('gpu/device', () => {
  it('exports the provider pair', () => {
    expect(typeof getDevice).toBe('function');
    expect(typeof setSharedDevice).toBe('function');
  });

  it('rejects when WebGPU is unavailable', async () => {
    await expect(getDevice()).rejects.toBeTruthy();
  });
});
