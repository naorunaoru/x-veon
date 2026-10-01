import type { EventEmitter } from 'node:events';

/** A utility process cannot be killed reliably until Electron emits spawn. */
export function waitForWorkerSpawn(
  child: EventEmitter & { readonly pid?: number },
  timeoutMs = 10_000,
): Promise<void> {
  if (child.pid !== undefined) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const finish = (error?: Error) => {
      clearTimeout(timer);
      child.removeListener('spawn', spawned);
      child.removeListener('exit', exited);
      if (error) reject(error);
      else resolve();
    };
    const spawned = () => finish();
    const exited = () => finish(Error('Worker exited before spawning'));
    const timer = setTimeout(
      () => finish(Error('Worker spawn timed out')),
      timeoutMs,
    );
    child.once('spawn', spawned);
    child.once('exit', exited);
  });
}
