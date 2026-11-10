import { beforeEach, describe, expect, it } from 'vitest';
import { StripPool, stripJobs, type StripJob } from './strip-pool';

const module = {} as WebAssembly.Module;

/** A fake strip worker: records what it is sent and answers each job as its script says. */
class FakeWorker {
  static created: FakeWorker[] = [];
  /** By strip index: an error to reply with, 'crash' for an error event, 'hang' for no reply. */
  static script = new Map<number, string>();
  onmessage: ((e: MessageEvent) => void) | null = null;
  onerror: ((e: ErrorEvent) => void) | null = null;
  terminated = false;
  inits: unknown[] = [];
  jobs: number[] = [];
  constructor() { FakeWorker.created.push(this); }
  postMessage(message: { module: WebAssembly.Module } | StripJob) {
    if ('module' in message) { this.inits.push(message.module); return; }
    this.jobs.push(message.index);
    const outcome = FakeWorker.script.get(message.index);
    if (outcome === 'hang') return;
    queueMicrotask(() => {
      if (this.terminated) return;
      if (outcome === 'crash') this.onerror?.({ message: 'unreachable', preventDefault() {} } as ErrorEvent);
      else if (outcome) this.onmessage?.({ data: { index: message.index, error: outcome } } as MessageEvent);
      else this.onmessage?.({ data: { index: message.index, block: new Uint16Array([message.index]) } } as MessageEvent);
    });
  }
  terminate() { this.terminated = true; }
}

const job = (index: number, bytes: number): StripJob => ({
  index, header: new Uint8Array(16), data: new Uint8Array(bytes), qBases: new Uint8Array(0), stripWidth: 768, cols: 768, height: 6,
});

const pool = (concurrency: number) => new StripPool(module, concurrency, () => new FakeWorker() as unknown as Worker);

describe('strip pool', () => {
  beforeEach(() => {
    FakeWorker.created = [];
    FakeWorker.script = new Map();
  });

  it('decodes every strip, one worker per strip when the cores allow, and keeps its workers', async () => {
    const strips = pool(12);
    const put: number[] = [];
    await strips.run([job(0, 10), job(1, 30), job(2, 20)], (index, block) => { put.push(index); expect(block[0]).toBe(index); });
    expect(put.sort()).toEqual([0, 1, 2]);
    expect(FakeWorker.created).toHaveLength(3);
    // Each worker got the compiled module once, before its first strip.
    expect(FakeWorker.created.map((w) => w.inits)).toEqual([[module], [module], [module]]);

    await strips.run([job(0, 10), job(1, 30), job(2, 20), job(3, 5)], () => {});
    expect(FakeWorker.created).toHaveLength(4);
    expect(FakeWorker.created.some((w) => w.terminated)).toBe(false);
  });

  it('hands the strips out largest first when there are fewer workers than strips', async () => {
    const strips = pool(2);
    const put: number[] = [];
    await strips.run([job(0, 10), job(1, 40), job(2, 30), job(3, 20)], (index) => { put.push(index); });
    expect(FakeWorker.created).toHaveLength(2);
    expect(FakeWorker.created.map((w) => w.jobs)).toEqual([[1, 3], [2, 0]]);
    expect(put.sort()).toEqual([0, 1, 2, 3]);
  });

  it('fails the whole decode on one strip error and terminates every worker', async () => {
    FakeWorker.script.set(1, 'RawLoaderError: "RAF compressed: strip data ended early"');
    FakeWorker.script.set(2, 'hang');
    const strips = pool(12);
    await expect(strips.run([job(0, 10), job(1, 30), job(2, 20)], () => {})).rejects.toThrow('strip data ended early');
    expect(FakeWorker.created.map((w) => w.terminated)).toEqual([true, true, true]);

    // The next decode starts from fresh workers.
    FakeWorker.script.clear();
    await strips.run([job(0, 10)], () => {});
    expect(FakeWorker.created).toHaveLength(4);
    expect(FakeWorker.created[3].terminated).toBe(false);
  });

  it('fails the decode when a strip worker crashes, or when a block is refused', async () => {
    FakeWorker.script.set(0, 'crash');
    const strips = pool(12);
    await expect(strips.run([job(0, 10), job(1, 30)], () => {})).rejects.toThrow('unreachable');
    expect(FakeWorker.created.every((w) => w.terminated)).toBe(true);

    FakeWorker.script.clear();
    await expect(strips.run([job(0, 10)], () => { throw new Error("A strip's block doesn't match its size"); })).rejects.toThrow("doesn't match");
    expect(FakeWorker.created.every((w) => w.terminated)).toBe(true);
  });
});

describe('strip jobs', () => {
  it("cuts each strip's bytes out of the file as the layout says", () => {
    const file = Uint8Array.from({ length: 64 }, (_, i) => i);
    // Header at 4, height 4038; strip 0: data 30..40, q_bases 20..24, width 768, 768 columns; strip 1: data 40..64, q_bases 24..28, width 672, 670 columns.
    const layout = new Uint32Array([4, 4038, 30, 40, 20, 24, 768, 768, 40, 64, 24, 28, 672, 670]);
    const jobs = stripJobs(file, layout);
    expect(jobs).toHaveLength(2);
    expect(jobs[0]).toMatchObject({ index: 0, stripWidth: 768, cols: 768, height: 4038 });
    expect(Array.from(jobs[0].header)).toEqual(Array.from({ length: 16 }, (_, i) => i + 4));
    expect(Array.from(jobs[0].data)).toEqual([30, 31, 32, 33, 34, 35, 36, 37, 38, 39]);
    expect(Array.from(jobs[0].qBases)).toEqual([20, 21, 22, 23]);
    expect(jobs[1]).toMatchObject({ index: 1, stripWidth: 672, cols: 670, height: 4038 });
    expect(jobs[1].data).toHaveLength(24);
    // Copies, so they can be transferred without detaching the file.
    expect(jobs[1].data.buffer).not.toBe(file.buffer);
    expect(jobs[0].data.buffer).not.toBe(jobs[1].data.buffer);
  });
});
