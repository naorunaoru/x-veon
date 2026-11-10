/// <reference lib="webworker" />
/**
 * Decodes single strips of a compressed Fuji RAF for the decode worker's strip pool. Its first
 * message is the decoder module the decode worker compiled; every later one is a strip, answered
 * with the strip's block as a transfer, or with an error. A panic traps this instance, and the
 * pool terminates all its workers on any error, so the instance is never used after one.
 */
import init, { decode_fuji_strip } from '../../../crates/rawloader-wasm/pkg/rawloader_wasm.js';
import type { StripInit, StripJob, StripReply } from './strip-pool';

let ready: Promise<unknown> | undefined;

self.onmessage = async (e: MessageEvent<StripInit | StripJob>) => {
  if ('module' in e.data) {
    ready = init({ module_or_path: e.data.module });
    // A failed start is reported with the first strip.
    ready.catch(() => {});
    return;
  }
  const { index, header, data, qBases, stripWidth, cols, height } = e.data;
  try {
    await ready;
    const block = decode_fuji_strip(header, data, qBases, stripWidth, cols, height);
    self.postMessage({ index, block } satisfies StripReply, [block.buffer]);
  } catch (err) {
    self.postMessage({ index, error: err instanceof Error ? err.message : String(err) } satisfies StripReply);
  }
};
