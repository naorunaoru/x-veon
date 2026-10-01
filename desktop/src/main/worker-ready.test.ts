import { EventEmitter } from 'node:events';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { waitForWorkerSpawn } from './worker-ready';

describe('utility worker readiness', () => {
  afterEach(() => vi.useRealTimers());

  it('waits for spawn before allowing a caller to restart the worker', async () => {
    const child = new EventEmitter();
    const ready = vi.fn();
    const pending = waitForWorkerSpawn(child).then(ready);
    await Promise.resolve();
    expect(ready).not.toHaveBeenCalled();
    child.emit('spawn');
    await pending;
    expect(ready).toHaveBeenCalledOnce();
    expect(child.listenerCount('exit')).toBe(0);
  });

  it('accepts an already spawned worker', async () => {
    const child = Object.assign(new EventEmitter(), { pid: 42 });
    await waitForWorkerSpawn(child);
    expect(child.eventNames()).toEqual([]);
  });

  it('rejects an exit before spawn and cleans up listeners', async () => {
    const child = new EventEmitter();
    const pending = waitForWorkerSpawn(child);
    child.emit('exit', 1);
    await expect(pending).rejects.toThrow('Worker exited before spawning');
    expect(child.eventNames()).toEqual([]);
  });

  it('bounds startup and removes its listeners on timeout', async () => {
    vi.useFakeTimers();
    const child = new EventEmitter();
    const pending = expect(waitForWorkerSpawn(child, 100)).rejects.toThrow('Worker spawn timed out');
    await vi.advanceTimersByTimeAsync(100);
    await pending;
    expect(child.eventNames()).toEqual([]);
  });
});
