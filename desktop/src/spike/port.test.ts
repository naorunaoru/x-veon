import { expect, it, vi } from 'vitest';
import { response } from './port';
class Port extends EventTarget {
  start() {}
  close() {}
}
it('rejects disconnected requests and removes listeners', async () => {
  const port = new Port() as unknown as MessagePort;
  const request = response(port, () => {});
  port.dispatchEvent(new Event('close'));
  await expect(request).rejects.toThrow('closed');
});
it('bounds a silent worker with a timeout', async () => {
  vi.useFakeTimers();
  try {
    const port = new Port() as unknown as MessagePort;
    const result = expect(response(port, () => {}, 10)).rejects.toThrow(
      'timed out',
    );
    await vi.advanceTimersByTimeAsync(10);
    await result;
  } finally {
    vi.useRealTimers();
  }
});
