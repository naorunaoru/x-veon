import path from 'node:path';
import { expect, it, vi } from 'vitest';
import { createExportTargets } from './exports';
const a = 'a'.repeat(22), b = 'b'.repeat(22), c = 'c'.repeat(22);
const raws: Record<string, string> = { [a]: '/photos/DSCF0001.RAF', [b]: '/photos/DSCF0002.RAF', [c]: '/other/DSCF0003.RAF' };
const folder = { isDirectory: () => true }, file = { isDirectory: () => false };
function harness(fixedDir?: string) {
  const deps = { rawPath: vi.fn((id: string) => raws[id]), register: vi.fn(async (_token: string, _dir: string) => {}), showItem: vi.fn(),
    chooseFolder: vi.fn(async () => '/chosen'), stat: vi.fn(async (_path: string) => folder), fixedDir };
  return { deps, targets: createExportTargets(deps) };
}
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
it('beside resolves the first RAW\'s folder and registers one token', async () => {
  const h = harness(); const result = await h.targets.resolve({ mode: 'beside' }, [a, b]);
  expect(result).toEqual({ token: expect.stringMatching(UUID) });
  expect(h.deps.register.mock.calls).toEqual([[result!.token, '/photos']]); expect(h.deps.stat).not.toHaveBeenCalled();
  await expect(h.targets.resolve({ mode: 'beside' }, ['d'.repeat(22)])).rejects.toThrow('Unknown photo');
  h.deps.register.mockRejectedValueOnce(new Error('worker stopped'));
  await expect(h.targets.resolve({ mode: 'beside' }, [a])).rejects.toThrow('worker stopped');
  h.targets.reveal(h.deps.register.mock.calls[1][0]); expect(h.deps.showItem).not.toHaveBeenCalled();
  expect(await h.targets.chooseFolder()).toBe('/chosen');
});
it('beside rejects photos from two folders', async () => {
  const h = harness();
  await expect(h.targets.resolve({ mode: 'beside' }, [a, c])).rejects.toThrow('Photos must be in one folder to export beside them'); expect(h.deps.register).not.toHaveBeenCalled();
});
it('folder re-validates the directory and reports the OS message', async () => {
  const h = harness(); const out = path.resolve('/out');
  const result = await h.targets.resolve({ mode: 'folder', path: out }, [a, c]);
  expect(h.deps.stat).toHaveBeenCalledWith(out); expect(h.deps.register).toHaveBeenCalledWith(result!.token, out);
  h.deps.stat.mockRejectedValueOnce(Object.assign(new Error("ENOENT: no such file or directory, stat '/gone'"), { code: 'ENOENT' }));
  await expect(h.targets.resolve({ mode: 'folder', path: '/gone' }, [a])).rejects.toThrow("Couldn't export to /gone: ENOENT: no such file or directory, stat '/gone'");
  h.deps.stat.mockResolvedValueOnce(file);
  await expect(h.targets.resolve({ mode: 'folder', path: '/file.txt' }, [a])).rejects.toThrow("Couldn't export to /file.txt: ");
  await expect(h.targets.resolve({ mode: 'folder', path: 'relative' }, [a])).rejects.toThrow("Couldn't export to relative: ");
  expect(h.deps.register).toHaveBeenCalledOnce();
});
it('fixedDir overrides saveTo', async () => {
  const h = harness('/fixed');
  const beside = await h.targets.resolve({ mode: 'beside' }, ['d'.repeat(22)]); const chosen = await h.targets.resolve({ mode: 'folder', path: '/out' }, [a]);
  expect(h.deps.register.mock.calls).toEqual([[beside!.token, '/fixed'], [chosen!.token, '/fixed']]); expect(h.deps.rawPath).not.toHaveBeenCalled(); expect(h.deps.stat).not.toHaveBeenCalled();
});
it('reveal shows the file when named and the folder itself otherwise', async () => {
  const h = harness(); const { token } = (await h.targets.resolve({ mode: 'beside' }, [a]))!;
  h.targets.reveal(token, 'DSCF0001.avif'); h.targets.reveal(token);
  h.targets.reveal('00000000-0000-4000-8000-000000000001', 'x.avif'); h.targets.reveal(token, '../escape.avif');
  expect(h.deps.showItem.mock.calls).toEqual([[path.join('/photos', 'DSCF0001.avif')], ['/photos']]);
});
it('keeps at most 50 tokens', async () => {
  const h = harness(); const tokens: string[] = [];
  for (let i = 0; i < 51; i++) tokens.push((await h.targets.resolve({ mode: 'beside' }, [a]))!.token);
  expect(new Set(tokens).size).toBe(51);
  h.targets.reveal(tokens[0]); expect(h.deps.showItem).not.toHaveBeenCalled();
  h.targets.reveal(tokens[1]); h.targets.reveal(tokens[50]); expect(h.deps.showItem).toHaveBeenCalledTimes(2);
});
it('rejects a UNC folder as an invalid target without calling stat', async () => {
  const h = harness();
  for (const unc of ['\\\\host\\share', '//host/share']) await expect(h.targets.resolve({ mode: 'folder', path: unc }, [a])).rejects.toThrow('Invalid export target');
  expect(h.deps.stat).not.toHaveBeenCalled(); expect(h.deps.register).not.toHaveBeenCalled();
});
it('rejects malformed ids and shapes', async () => {
  const h = harness('/fixed');
  for (const [saveTo, ids] of [[{ mode: 'beside' }, []], [{ mode: 'beside' }, ['../escape']], [{ mode: 'beside' }, Array(251).fill(a)], [{ mode: 'folder' }, [a]], [{ mode: 'folder', path: '' }, [a]],
    [{ mode: 'folder', path: '/a\0b' }, [a]], [{ mode: 'folder', path: '/' + 'x'.repeat(4096) }, [a]], [{ mode: 'desktop' }, [a]], [null, [a]], [{ mode: 'beside' }, 'aaaa']] as const)
    await expect(h.targets.resolve(saveTo as never, ids as never)).rejects.toThrow('Invalid export target');
  expect(h.deps.register).not.toHaveBeenCalled();
});
