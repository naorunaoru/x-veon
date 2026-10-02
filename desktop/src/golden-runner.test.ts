import { afterEach, expect, it, vi } from 'vitest';
const m = vi.hoisted(() => ({ start: vi.fn(), discard: vi.fn(), remove: vi.fn(), run: vi.fn() }));
vi.mock('@/startApp', () => ({ startApp: m.start }));
vi.mock('@/dev/golden', () => ({ runGolden: m.run }));
vi.mock('@/app/services/processing', () => ({ discardResult: m.discard }));
vi.mock('@/app/store', () => ({ useAppStore: { getState: () => ({ removeFile: m.remove }) } }));
afterEach(() => { vi.unstubAllGlobals(); vi.clearAllMocks(); });
it('starts the golden host, cleans fixture ownership, and publishes a terminal report and elapsed time', async () => {
  vi.stubGlobal('window', {}); vi.stubGlobal('location', { search: '?golden=render' });
  const module = await import('./golden-runner').catch(() => null); expect(module?.runGoldenApp).toBeTypeOf('function');
  const root = {} as HTMLElement;
  m.run.mockImplementationOnce(async (cleanup: (id: string) => Promise<void>) => {
    const host = m.start.mock.calls[0][1]; const result = await host.library.addFiles([new File(['raw'], 'DSCF3332.RAF')]); const id = result.photos[0].id;
    await cleanup(id); expect(m.discard).toHaveBeenCalledWith(id); expect(m.remove).toHaveBeenCalledWith(id); await expect(host.library.readRaw(id)).rejects.toThrow('Unknown fixture');
    (window as any).__golden = { status: 'PASS', results: [1] };
  });
  await module!.runGoldenApp(root);
  expect(m.start).toHaveBeenCalledWith(root, expect.objectContaining({ library: expect.any(Object), settingsDbName: 'xveon-desktop-golden' }));
  expect((window as any).__goldenRun).toEqual({ status: 'PASS', totalMs: expect.any(Number), report: { status: 'PASS', results: [1] } }); expect((window as any).__goldenRun.totalMs).toBeGreaterThanOrEqual(0);
});
