import { createHash } from 'node:crypto';
import path from 'node:path';
import { directoryIdentity, writeFileAtomic, type DirectoryIdentity } from './atomic-write';
import type { NativeLoad } from './native';
import type { ExportAvailability, ExportBegin, ExportReceipt } from '../protocol/rpc';

type Job = {
  begin: ExportBegin; target: string; directory?: DirectoryIdentity;
  planes: Float32Array[]; received: number[];
  state: 'pending' | 'receiving' | 'queued' | 'encoding' | 'writing';
  abort: AbortController; release?: () => void;
};
export function createExportService(opts: { native: () => NativeLoad; now?: () => number; slots?: number }) {
  const now = opts.now ?? (() => performance.now());
  const destinations = new Map<string, string>();
  const jobs = new Map<string, Job>();
  let stopping = false;
  const writes = new Set<Promise<void>>();
  let loaded: NativeLoad | undefined;
  const native = () => (loaded ??= opts.native());
  let free = opts.slots ?? 2;
  if (!Number.isSafeInteger(free) || free < 1) throw new Error('Invalid export slot count');
  const waiting: (() => void)[] = [];
  function acquire(signal: AbortSignal): Promise<() => void> {
    signal.throwIfAborted();
    return new Promise((resolve, reject) => {
      const cancel = () => {
        const index = waiting.indexOf(grant);
        if (index !== -1) waiting.splice(index, 1);
        reject(signal.reason);
      };
      const grant = () => {
        signal.removeEventListener('abort', cancel);
        let released = false;
        resolve(() => {
          if (released) return;
          released = true;
          const next = waiting.shift();
          if (next) next(); else free++;
        });
      };
      if (free > 0) { free--; grant(); }
      else { waiting.push(grant); signal.addEventListener('abort', cancel, { once: true }); }
    });
  }
  let chain: Promise<unknown> = Promise.resolve();
  function job(id: string) { const j = jobs.get(id); if (!j) throw new Error('Unknown export job'); return j; }
  function drop(j: Job) {
    if (jobs.get(j.begin.job) === j) jobs.delete(j.begin.job);
    j.planes = []; j.release?.();
  }
  function cancel(id: string) {
    const j = jobs.get(id);
    if (!j) return;
    j.abort.abort(new Error('Export cancelled'));
    // Native encode borrows the arrays until its promise settles. Keep the slot
    // and their references alive, without mutating or detaching their buffers.
    if (j.state !== 'encoding' && j.state !== 'writing') drop(j);
  }
  return {
    status(): ExportAvailability { if (stopping) return { available: false, reason: 'The background worker stopped.' }; const n = native(); return n.ok ? { available: true } : { available: false, reason: n.reason }; },
    register(token: string, target: string) {
      if (stopping) throw new Error('The background worker stopped.');
      if (!path.isAbsolute(target)) { console.error('Ignored a relative export destination'); return; }
      destinations.set(token, target);
    },
    async begin(message: ExportBegin) {
      if (stopping) throw new Error('The background worker stopped.');
      const n = native(); if (!n.ok) throw new Error(n.reason);
      if (jobs.has(message.job)) throw new Error('Duplicate export job');
      const target = destinations.get(message.destination);
      if (!target) throw new Error('Unknown or used export destination');
      destinations.delete(message.destination);
      const j: Job = { begin: message, target, planes: [], received: [], state: 'pending', abort: new AbortController() };
      // Reserve the ID before directory lookup or slot acquisition can yield.
      jobs.set(message.job, j);
      try {
        j.directory = await directoryIdentity(path.dirname(target));
        j.abort.signal.throwIfAborted();
        j.release = await acquire(j.abort.signal);
        j.abort.signal.throwIfAborted();
        const length = message.width * message.height * 3;
        j.planes = Array.from({ length: message.planes }, () => new Float32Array(length));
        j.received = Array(message.planes).fill(0);
        j.state = 'receiving';
      } catch (error) { drop(j); if (j.abort.signal.aborted) throw j.abort.signal.reason; throw error; }
    },
    chunk(id: string, plane: 0 | 1, offset: number, data: ArrayBuffer) {
      const j = job(id); if (j.state !== 'receiving') throw new Error('Export data after commit');
      const target = j.planes[plane]; if (!target) throw new Error('Invalid export plane');
      if (offset !== j.received[plane]) throw new Error('Export data out of order');
      if (offset + data.byteLength > target.byteLength) throw new Error('Export data overflows its plane');
      new Uint8Array(target.buffer, offset, data.byteLength).set(new Uint8Array(data));
      j.received[plane] += data.byteLength;
    },
    async commit(id: string): Promise<ExportReceipt> {
      const j = job(id);
      if (j.state === 'pending') throw new Error('Incomplete export data');
      if (j.state !== 'receiving') throw new Error('Export already committed');
      if (j.received.some((bytes, i) => bytes !== j.planes[i].byteLength)) throw new Error('Incomplete export data');
      j.state = 'queued';
      const run = chain.then(async () => {
        j.abort.signal.throwIfAborted();
        const n = native(); if (!n.ok) throw new Error(n.reason);
        j.state = 'encoding';
        const { format, width, height, orientation, quality, peakLuminance } = j.begin;
        const started = now();
        const bytes = await n.encoder.encode(j.planes[0], j.planes[1] ?? null, { format, width, height, orientation, quality, peakLuminance });
        const encodeMs = now() - started;
        j.planes = [];
        j.abort.signal.throwIfAborted();
        j.state = 'writing';
        const write = writeFileAtomic(j.target, bytes, { directory: j.directory, signal: j.abort.signal });
        writes.add(write);
        try { await write; }
        catch (cause) {
          if (cause === j.abort.signal.reason) throw cause;
          const error = cause as NodeJS.ErrnoException;
          // Node's message can contain the random temp path. Keep useful OS detail
          // and the original cause while naming the user's actual destination.
          const detail = (cause instanceof Error ? cause.message : String(cause)).replace(/\.[^/\\]*\.[a-f0-9-]{36}\.tmp/g, path.basename(j.target));
          throw Object.assign(new Error(`Couldn't write ${path.basename(j.target)}: ${detail}`, { cause }), { code: error.code });
        }
        finally { writes.delete(write); }
        return { name: path.basename(j.target), bytes: bytes.byteLength, sha256: createHash('sha256').update(bytes).digest('hex'), encodeMs };
      });
      chain = run.catch(() => {});
      try { return await run; } finally { drop(j); }
    },
    async shutdown() {
      stopping = true;
      destinations.clear();
      for (const id of [...jobs.keys()]) cancel(id);
      // Cancellation prevents encodes from starting any later write. Native code
      // may still borrow its planes, so only wait for existing atomic cleanup.
      await Promise.allSettled([...writes]);
    },
    cancel,
    cancelAll() { for (const id of [...jobs.keys()]) cancel(id); },
  };
}
