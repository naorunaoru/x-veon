import { expect, it, vi } from 'vitest';
import { createExportDestinations } from './exports';
const id = 'a'.repeat(22);
function harness(fixedDir?: string) {
  const deps = { showSaveDialog: vi.fn(async (_options: Electron.SaveDialogOptions) => ({ canceled: false, filePath: '/chosen/DSCF3332.avif' })), rawPath: vi.fn((photo: string) => photo === id ? '/photos/DSCF3332.RAF' : undefined), register: vi.fn(async (_token: string, _target: string) => {}), reveal: vi.fn(), fallbackDir: () => '/pictures', fixedDir };
  return { deps, exports: createExportDestinations(deps) };
}
it('suggests the RAW basename and catalogue filter, preserving the chosen OS path', async () => {
  const h = harness(); const result = await h.exports.choose(id, 'avif');
  expect(h.deps.showSaveDialog).toHaveBeenCalledWith({ defaultPath: '/photos/DSCF3332.avif', filters: [{ name: 'AVIF (BT.2020 / HLG)', extensions: ['avif'] }], properties: ['createDirectory', 'showOverwriteConfirmation'] });
  expect(result).toEqual({ token: expect.stringMatching(/^[0-9a-f-]{36}$/), name: 'DSCF3332.avif' });
  expect(h.deps.register).toHaveBeenCalledWith(result!.token, '/chosen/DSCF3332.avif');
  await h.exports.choose('b'.repeat(22), 'avif'); expect(h.deps.showSaveDialog.mock.calls[1][0].defaultPath).toBe('/pictures/export.avif');
});
it('cancels without registration and issues unique tokens with a 50 entry reveal bound', async () => {
  const h = harness(); h.deps.showSaveDialog.mockResolvedValueOnce({ canceled: true, filePath: '' });
  expect(await h.exports.choose(id, 'avif')).toBeNull(); expect(h.deps.register).not.toHaveBeenCalled();
  const results = []; for (let i = 0; i < 51; i++) results.push((await h.exports.choose(id, 'avif'))!);
  expect(new Set(results.map(r => r.token)).size).toBe(51);
  h.exports.reveal(results[0].token); h.exports.reveal('unknown'); expect(h.deps.reveal).not.toHaveBeenCalled();
  h.exports.reveal(results[50].token); expect(h.deps.reveal).toHaveBeenCalledWith('/chosen/DSCF3332.avif');
});
it('waits for registration ACK and exposes no reveal record on failure', async () => {
  const h = harness(); let reject!: (error: Error) => void;
  h.deps.register.mockImplementationOnce(() => new Promise((_resolve, fail) => { reject = fail; }));
  const pending = h.exports.choose(id, 'avif'); const settled = vi.fn(); void pending.then(settled, settled);
  await vi.waitFor(() => expect(h.deps.register).toHaveBeenCalled());
  const token = h.deps.register.mock.calls[0][0]; h.exports.reveal(token); expect(h.deps.reveal).not.toHaveBeenCalled(); expect(settled).not.toHaveBeenCalled();
  reject(new Error('worker stopped')); await expect(pending).rejects.toThrow('worker stopped');
  h.exports.reveal(token); expect(h.deps.reveal).not.toHaveBeenCalled();
});
it('uses a safe leaf in a fixed directory and rejects direct traversal calls', async () => {
  const h = harness('/fixed'); const result = await h.exports.choose(id, 'avif');
  expect(h.deps.showSaveDialog).not.toHaveBeenCalled(); expect(h.deps.register).toHaveBeenCalledWith(result!.token, `/fixed/${id}-avif.avif`);
  await expect(h.exports.choose('../escape', 'avif')).rejects.toThrow();
  await expect(h.exports.choose(id, '../escape' as never)).rejects.toThrow();
});
