import { afterEach, expect, it, vi } from 'vitest';
import type { DesktopBridge } from '../protocol/bridge';
import { waitForWorkerPort } from './port';
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });
it('aborting a pending handshake removes its listener and timer and observes a later invoke rejection', async () => {
  vi.useFakeTimers(); const win = new EventTarget(); vi.stubGlobal('window', win); vi.stubGlobal('location', { origin: 'app://bundle' });
  const removed = vi.spyOn(win, 'removeEventListener');
  let rejectInvoke!: (error: Error) => void;
  const bridge = { requestWorkerPort: vi.fn(() => new Promise<void>((_resolve, reject) => { rejectInvoke = reject; })) } as unknown as DesktopBridge;
  const abort = new AbortController(); const pending = waitForWorkerPort(bridge, abort.signal);
  const failed = expect(pending).rejects.toThrow('Worker restarted'); abort.abort(new Error('Worker restarted')); await failed;
  expect(removed).toHaveBeenCalledWith('message', expect.any(Function)); expect(vi.getTimerCount()).toBe(0);
  rejectInvoke(new Error('old main request failed')); await Promise.resolve(); await Promise.resolve();
});
it('an already cancelled handshake never requests or listens for a port', async () => {
  const win = new EventTarget(); vi.stubGlobal('window', win); const added = vi.spyOn(win, 'addEventListener');
  const bridge = { requestWorkerPort: vi.fn() } as unknown as DesktopBridge;
  const abort = new AbortController(); abort.abort(new Error('Worker stopped'));
  await expect(waitForWorkerPort(bridge, abort.signal)).rejects.toThrow('Worker stopped');
  expect(bridge.requestWorkerPort).not.toHaveBeenCalled(); expect(added).not.toHaveBeenCalled();
});
