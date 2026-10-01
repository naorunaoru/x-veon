import fs from 'node:fs';
import { EventEmitter } from 'node:events';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { createWatchHandler, watchFolder } from './watch';
let sources: EventEmitter[];
beforeEach(() => {
  vi.useFakeTimers(); sources = [];
  vi.spyOn(fs, 'watch').mockImplementation(((...args: unknown[]) => {
    const source = new EventEmitter();
    source.on('change', args.at(-1) as (...args: unknown[]) => void);
    sources.push(source);
    return Object.assign(source, { close() { source.removeAllListeners(); }, ref() { return this; }, unref() { return this; } });
  }) as typeof fs.watch);
});
afterEach(() => { vi.useRealTimers(); vi.restoreAllMocks(); });
it('emits one replacement only after 200ms without another notification in a 50-event burst', async () => {
  const changed = vi.fn(); const watcher = watchFolder('/photos', changed);
  for (let i = 0; i < 50; i++) {
    sources[0].emit('change', 'rename', `${i}.RAF`);
    await vi.advanceTimersByTimeAsync(19);
  }
  expect(changed).not.toHaveBeenCalled();
  await vi.advanceTimersByTimeAsync(180); expect(changed).not.toHaveBeenCalled();
  await vi.advanceTimersByTimeAsync(1); expect(changed).toHaveBeenCalledOnce();
  await vi.advanceTimersByTimeAsync(1000); expect(changed).toHaveBeenCalledOnce();
  watcher.close();
});
it('cancels the old folder timer when changing watches and prevents callbacks after close', async () => {
  const changed = vi.fn(); const watch = createWatchHandler(changed);
  watch({ path: '/A' }); sources[0].emit('change'); await vi.advanceTimersByTimeAsync(100);
  watch({ path: '/B' }); await vi.advanceTimersByTimeAsync(100); expect(changed).not.toHaveBeenCalled();
  sources[1].emit('change'); await vi.advanceTimersByTimeAsync(200); expect(changed).toHaveBeenCalledOnce();
  sources[1].emit('change'); watch(null); await vi.advanceTimersByTimeAsync(200); expect(changed).toHaveBeenCalledOnce();
});
