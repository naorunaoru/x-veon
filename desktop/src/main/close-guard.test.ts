import { EventEmitter } from 'node:events';
import { afterEach, expect, it, vi } from 'vitest';
import { createCloseGuard, unsavedQuitMessage } from './close-guard';
import type { UnsavedSummary } from '@/host';
const unsaved: UnsavedSummary[] = [{ id: 'a', name: 'one', folder: { id: 'a', name: 'Alps' }, error: null }, { id: 'b', name: 'two', folder: { id: 'b', name: 'Beach' }, error: 'locked' }];
afterEach(() => vi.useRealTimers());
function setup(flush = () => Promise.resolve([] as UnsavedSummary[]), answer = true) {
  const window = Object.assign(new EventEmitter(), { destroy: vi.fn() });
  const app = Object.assign(new EventEmitter(), { quit: vi.fn(() => { app.emit('before-quit', { preventDefault: passed }); }), exit: vi.fn() });
  const passed = vi.fn(), confirm = vi.fn(async () => answer), requestFlush = vi.fn(flush);
  const guard = createCloseGuard({ window, app, inventory: () => unsaved, requestFlush, confirmQuit: confirm, timeoutMs: 20 });
  return { window, app, passed, confirm, requestFlush, guard };
}
it.each(['close', 'before-quit', 'close'])('prevents %s synchronously and quits after clean flush', async event => {
  const h = setup(); const preventDefault = vi.fn(); (event === 'close' ? h.window : h.app).emit(event, { preventDefault });
  expect(preventDefault).toHaveBeenCalledOnce(); expect(h.app.quit).not.toHaveBeenCalled(); await vi.waitFor(() => expect(h.app.quit).toHaveBeenCalledOnce()); expect(h.passed).not.toHaveBeenCalled(); expect(h.window.destroy).not.toHaveBeenCalled();
});
it('Cancel retains renderer and another close can retry', async () => {
  const h = setup(async () => unsaved, false); h.window.emit('close', { preventDefault() {} }); await vi.waitFor(() => expect(h.confirm).toHaveBeenCalledWith(unsaved));
  expect(h.app.quit).not.toHaveBeenCalled(); expect(h.window.destroy).not.toHaveBeenCalled();
  h.app.emit('before-quit', { preventDefault() {} }); await vi.waitFor(() => expect(h.requestFlush).toHaveBeenCalledTimes(2));
  expect(unsavedQuitMessage(unsaved)).toContain('Alps'); expect(unsavedQuitMessage(unsaved)).toContain('Beach'); expect(unsavedQuitMessage(unsaved)).toContain('2 photos');
});
it('uses pushed inventory on timeout, shares concurrent flows and permits Quit anyway', async () => {
  vi.useFakeTimers(); const h = setup(() => new Promise(() => {}));
  h.window.emit('close', { preventDefault() {} }); h.app.emit('before-quit', { preventDefault() {} });
  await vi.advanceTimersByTimeAsync(20); expect(h.requestFlush).toHaveBeenCalledOnce(); expect(h.confirm).toHaveBeenCalledWith(unsaved); expect(h.app.quit).toHaveBeenCalledOnce(); expect(h.passed).not.toHaveBeenCalled();
});
it('shutdown flushes best effort then exits without a dialog', async () => {
  vi.useFakeTimers(); const h = setup(() => new Promise(() => {})); h.guard.onSessionEnd(); await vi.advanceTimersByTimeAsync(20); expect(h.app.exit).toHaveBeenCalledWith(0); expect(h.confirm).not.toHaveBeenCalled();
});

it('names every unsaved photo beside its folder', () => {
  expect(unsavedQuitMessage(unsaved)).toBe(
    "2 photos have edits that aren't saved.\n\none — Alps\ntwo — Beach",
  );
});
it('uses singular grammar and names the photo when its folder is unknown', () => {
  expect(unsavedQuitMessage([{ id: 'x', name: 'DSCF3332.RAF', folder: null, error: null }])).toBe(
    "1 photo has edits that aren't saved.\n\nDSCF3332.RAF — Unknown folder",
  );
});
it('keeps identically named photos from different folders in the warning', () => {
  const photos = unsaved.map(photo => ({ ...photo, name: 'same.RAF' }));
  expect(unsavedQuitMessage(photos)).toBe(
    "2 photos have edits that aren't saved.\n\nsame.RAF — Alps\nsame.RAF — Beach",
  );
});

it('bounds warning names at ten without changing the full unsaved inventory', () => {
 const all = Array.from({ length: 350 }, (_, i) => ({ ...unsaved[0], id: String(i), name: `photo-${i}` }));
 const text = unsavedQuitMessage(all);
 expect(text).toContain('350 photos'); expect(text).toContain('photo-9'); expect(text).not.toContain('photo-10'); expect(text).toContain('340 more'); expect(all).toHaveLength(350);
});
