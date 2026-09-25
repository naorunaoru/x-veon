import { beforeEach, describe, expect, it } from 'vitest';
import { decodeRaw, initWasm, setDecodeWorkerFactory } from './raf-decoder';

/** A fake decode worker: answers each request with the next scripted reply, asynchronously. */
class FakeWorker {
  static created: FakeWorker[] = [];
  onmessage: ((e: MessageEvent) => void) | null = null;
  onerror: ((e: ErrorEvent) => void) | null = null;
  terminated = false;
  constructor(private replies: Array<object | 'crash'>) { FakeWorker.created.push(this); }
  postMessage(message: { type: string }) {
    const reply = this.replies.shift() ?? { type: message.type === 'ping' ? 'pong' : 'done', raw: { width: 1 } };
    queueMicrotask(() => {
      if (reply === 'crash') this.onerror?.({ message: 'unreachable', preventDefault() {} } as ErrorEvent);
      else this.onmessage?.({ data: reply } as MessageEvent);
    });
  }
  terminate() { this.terminated = true; }
}

describe('RAW decoder worker', () => {
  let script: Array<Array<object | 'crash'>>;
  beforeEach(() => {
    FakeWorker.created = [];
    script = [];
    setDecodeWorkerFactory(() => new FakeWorker(script.shift() ?? []) as unknown as Worker);
  });

  it('decodes in one worker while files succeed', async () => {
    await initWasm();
    await decodeRaw(new ArrayBuffer(8));
    await decodeRaw(new ArrayBuffer(8));
    expect(FakeWorker.created).toHaveLength(1);
  });

  it('replaces the worker after a crash or a decode error, so later files still decode', async () => {
    script = [['crash'], [{ type: 'error', message: 'RawLoaderError: truncated' }], []];
    await expect(decodeRaw(new ArrayBuffer(8))).rejects.toThrow('unreachable');
    await expect(decodeRaw(new ArrayBuffer(8))).rejects.toThrow('truncated');
    await expect(decodeRaw(new ArrayBuffer(8))).resolves.toMatchObject({ width: 1 });
    expect(FakeWorker.created.map((w) => w.terminated)).toEqual([true, true, false]);
  });

  it('runs decodes one at a time', async () => {
    const results = await Promise.all([decodeRaw(new ArrayBuffer(1)), decodeRaw(new ArrayBuffer(1))]);
    expect(results).toHaveLength(2);
    expect(FakeWorker.created).toHaveLength(1);
  });
});
