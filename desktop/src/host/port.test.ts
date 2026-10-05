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

it('returns receipt, availability and stamp and applies each request timeout independently', async () => {
  vi.useFakeTimers();
  const win = new EventTarget(); vi.stubGlobal('window', win); vi.stubGlobal('location', { origin: 'app://bundle' });
  class Port extends EventTarget {
    sent: { rid: number }[] = [];
    start() {} close() {}
    postMessage(request: { rid: number }, transfer?: unknown) { expect(transfer).toBeUndefined(); this.sent.push(request); }
  }
  const port = new Port();
  const bridge = { requestWorkerPort: vi.fn(async requestId => {
    const event = new MessageEvent('message', { data: { type: 'xveon-port', version: 2, requestId }, origin: 'app://bundle' });
    Object.defineProperties(event, { source: { value: win }, ports: { value: [port] } }); win.dispatchEvent(event);
  }) } as unknown as DesktopBridge;
  const { createWorkerClient } = await import('./port');
  const acknowledged = vi.fn(); const client = createWorkerClient(bridge, () => {}, acknowledged);
  const short = client.request({ op: 'exportStatus' }, { timeoutMs: 100 });
  const long = client.request({ op: 'exportStatus' }); let longSettled = false;
  void long.then(() => { longSettled = true; });
  const failed = expect(short).rejects.toThrow('Worker response timed out');
  await vi.advanceTimersByTimeAsync(100); await failed; expect(longSettled).toBe(false);
  const reply = { v: 1, rid: port.sent[1].rid, ok: true, availability: { available: true },
    stamp: { worker: '11111111-1111-4111-8111-111111111111', revision: 2 },
    receipt: { bytes: 20, sha256: 'a'.repeat(64), encodeMs: 12, name: 'x.avif' } };
  port.dispatchEvent(new MessageEvent('message', { data: reply }));
  expect(await long).toEqual(reply); expect(acknowledged).toHaveBeenCalledWith(reply.stamp); expect(vi.getTimerCount()).toBe(0);
});

it('aborts a request awaiting initial delivery without cancelling the shared connection', async () => {
  vi.useFakeTimers();
  const win = new EventTarget(); vi.stubGlobal('window', win); vi.stubGlobal('location', { origin: 'app://bundle' });
  const requests: string[] = [];
  const bridge = { requestWorkerPort: vi.fn(async (requestId: string) => { requests.push(requestId); }) } as unknown as DesktopBridge;
  const { createWorkerClient } = await import('./port');
  const client = createWorkerClient(bridge, () => {}), abort = new AbortController();
  const rejected = vi.fn();
  const cancelled = client.request({ op: 'exportStatus' }, { signal: abort.signal }).catch(rejected);
  const other = client.request({ op: 'rescan' });
  abort.abort();
  for (let i = 0; i < 20; i++) await Promise.resolve();
  expect(rejected).toHaveBeenCalledWith(expect.objectContaining({ name: 'AbortError' }));
  expect(vi.getTimerCount()).toBe(1); // The shared handshake remains alive.
  class Port extends EventTarget {
    sent: string[] = [];
    start() {} close() {}
    postMessage(request: { op: string; rid: number }) {
      this.sent.push(request.op);
      queueMicrotask(() => this.dispatchEvent(new MessageEvent('message', { data: { v: 1, rid: request.rid, ok: true } })));
    }
  }
  const port = new Port();
  const event = new MessageEvent('message', { data: { type: 'xveon-port', version: 2, requestId: requests[0] }, origin: 'app://bundle' });
  Object.defineProperties(event, { source: { value: win }, ports: { value: [port] } }); win.dispatchEvent(event);
  await other; await cancelled;
  expect(port.sent).toEqual(['rescan']); expect(bridge.requestWorkerPort).toHaveBeenCalledTimes(1); expect(vi.getTimerCount()).toBe(0);
});
