import { expect, it, vi } from 'vitest';
const state = vi.hoisted(() => ({ options: undefined as any, open: undefined as any, navigate: undefined as any, permission: undefined as any, check: undefined as any, external: vi.fn(async (_url: string) => {}) }));
vi.mock('electron', () => ({
  BrowserWindow: class { constructor(opts: unknown) { state.options = opts; } webContents = { setWindowOpenHandler: (h: unknown) => { state.open = h; }, on: (_event: string, h: unknown) => { state.navigate = h; } }; },
  session: { defaultSession: { setPermissionRequestHandler: (h: unknown) => { state.permission = h; }, setPermissionCheckHandler: (h: unknown) => { state.check = h; } } }, shell: { openExternal: state.external },
}));
import { createMainWindow } from './window';
it('keeps the renderer sandboxed, denies permissions/navigation and allows only external HTTP(S)', () => {
  createMainWindow({ preload: '/preload.js' });
  expect(state.options.webPreferences).toMatchObject({ sandbox: true, contextIsolation: true, nodeIntegration: false, backgroundThrottling: false, preload: '/preload.js' });
  const preventDefault = vi.fn(), permission = vi.fn(); state.navigate({ preventDefault }); expect(preventDefault).toHaveBeenCalledOnce();
  state.permission(null, 'camera', permission); expect(permission).toHaveBeenCalledWith(false); expect(state.check()).toBe(false);
  for (const url of ['https://example.com', 'http://example.com', 'file:///secret', 'javascript:alert(1)', 'xveon-photo://raw/abc', 'invalid']) expect(state.open({ url })).toEqual({ action: 'deny' });
  expect(state.external.mock.calls.map(c => c[0])).toEqual(['https://example.com', 'http://example.com']);
});
