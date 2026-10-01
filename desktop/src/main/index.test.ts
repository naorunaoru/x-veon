import { afterEach, expect, it, vi } from 'vitest';
const state = vi.hoisted(() => ({ boot: undefined as Promise<unknown> | undefined, handles: new Map<string, any>(), listeners: new Map<string, any>(), privileges: vi.fn(), name: vi.fn(), quit: vi.fn(), stop: vi.fn(), win: undefined as any, app: undefined as any, report: vi.fn(async () => {}), workerEvent: undefined as any, restart: vi.fn(async () => {}) }));
vi.mock('electron', async () => {
  const { EventEmitter } = await import('node:events');
  const app = Object.assign(new EventEmitter(), { setName: state.name, setPath: vi.fn(), getPath: () => '/tmp/task9-userdata', whenReady: () => ({ then: (fn: () => Promise<unknown>) => { state.boot = fn(); return state.boot; } }), quit: state.quit, exit: vi.fn() }); state.app = app;
  const win = Object.assign(new EventEmitter(), { webContents: Object.assign(new EventEmitter(), { send: vi.fn(), setWindowOpenHandler: vi.fn(), mainFrame: {}, executeJavaScript: vi.fn() }), loadURL: vi.fn(async () => {}), destroy: vi.fn(), isDestroyed: () => false }); state.win = win;
  return { app, BrowserWindow: class { constructor() { return win; } }, powerMonitor: new EventEmitter(), screen: {}, ipcMain: { handle: (name: string, fn: any) => state.handles.set(name, fn), on: (name: string, fn: any) => state.listeners.set(name, fn) }, MessageChannelMain: class {}, net: {}, protocol: { registerSchemesAsPrivileged: state.privileges, handle: vi.fn() }, session: { defaultSession: { setPermissionRequestHandler: vi.fn(), setPermissionCheckHandler: vi.fn() } }, utilityProcess: {}, Menu: { buildFromTemplate: (t: unknown) => t, setApplicationMenu: vi.fn() }, dialog: { showOpenDialog: vi.fn(async () => ({ canceled: true, filePaths: [] })), showErrorBox: vi.fn(), showMessageBox: vi.fn(async () => ({ response: 1 })) }, shell: {} };
});
vi.mock('node:fs/promises', () => ({ readdir: vi.fn(async () => []), mkdir: vi.fn(async () => {}), writeFile: vi.fn(async () => {}) }));
vi.mock('./folders', () => ({ createFolderStore: () => ({ recent: () => [], last: () => null, resolve: () => null }), folderId: () => 'f' }));
vi.mock('./worker', () => ({ createWorkerSupervisor: (opts: any) => { state.workerEvent = opts.onEvent; return ({ restart: state.restart, roots: [], registry: new Map(), onMessage() {}, stop: state.stop, request: vi.fn(), connect: vi.fn() }); } }));
vi.mock('./golden-report', () => ({ watchGoldenReport: state.report }));
const originalArgs = [...process.argv];
afterEach(async () => { const { dialog, powerMonitor } = await import('electron'); state.app?.removeAllListeners(); state.win?.removeAllListeners(); state.win?.webContents.removeAllListeners(); powerMonitor.removeAllListeners(); state.quit.mockReset(); vi.mocked(dialog.showMessageBox).mockReset().mockResolvedValue({ response: 1, checkboxChecked: false }); process.argv = [...originalArgs]; vi.useRealTimers(); vi.unstubAllGlobals(); vi.clearAllMocks(); state.handles.clear(); state.listeners.clear(); });
it('boots with only the desktop route, registers a secure streaming scheme, and defers worker cleanup until quit is accepted', async () => {
  process.argv.push('--golden-report=/ignored.json');
  vi.useFakeTimers(); vi.stubGlobal('__XV_GOLDEN__', false); await import('./index'); await state.boot;
  expect([...state.handles.keys()]).toEqual(['xveon-desktop']); expect(state.handles.has('xveon-desktop')).toBe(true);
  expect(state.privileges.mock.calls[0][0]).toContainEqual({ scheme: 'xveon-photo', privileges: { standard: true, secure: true, supportFetchAPI: true, corsEnabled: true, stream: true } });
  expect(state.name).toHaveBeenCalledWith('X-veon Dev'); expect(state.win.loadURL).toHaveBeenCalledWith('app://bundle/?');
  const preventDefault = vi.fn(); state.app.emit('before-quit', { preventDefault }); expect(preventDefault).toHaveBeenCalledOnce(); expect(state.stop).not.toHaveBeenCalled();
  state.app.emit('will-quit'); expect(state.stop).toHaveBeenCalledOnce();
});

it('loads the render golden route and starts the reporter only in a golden build', async () => {
  vi.resetModules(); vi.stubGlobal('__XV_GOLDEN__', true); process.argv.push('--golden-report=/report.json');
  await import('./index'); await state.boot;
  expect(state.win.loadURL).toHaveBeenCalledWith('app://bundle/?golden=render'); expect(state.report).toHaveBeenCalledWith(state.win, '/report.json');
});

it('offers native Restart and reconnects after the crash limit', async () => {
  vi.resetModules(); vi.stubGlobal('__XV_GOLDEN__', false);
  const { dialog } = await import('electron');
  vi.mocked(dialog.showMessageBox).mockResolvedValueOnce({ response: 0, checkboxChecked: false });
  await import('./index'); await state.boot;
  state.workerEvent({ stopped: 'The background worker stopped.' });
  await vi.waitFor(() => expect(state.restart).toHaveBeenCalledOnce());
  expect(dialog.showMessageBox).toHaveBeenCalledWith(state.win, expect.objectContaining({ buttons: ['Restart', 'Quit'] }));
  expect(state.win.webContents.send).toHaveBeenCalledWith('xveon-event', expect.objectContaining({ kind: 'worker-stopped' }));
});
it.each([false, true])('offers Restart after cancelling crash-dialog Quit unless shutdown is ending: %s', async ending => {
  vi.resetModules(); vi.useFakeTimers(); vi.stubGlobal('__XV_GOLDEN__', false);
  const { dialog } = await import('electron');
  let cancel!: (value: Electron.MessageBoxReturnValue) => void;
  vi.mocked(dialog.showMessageBox).mockResolvedValueOnce({ response: 1, checkboxChecked: false })
    .mockImplementationOnce(() => new Promise(resolve => { cancel = resolve; }))
    .mockImplementationOnce(() => new Promise(() => {}));
  await import('./index'); await state.boot;
  const event = { sender: state.win.webContents, senderFrame: Object.assign(state.win.webContents.mainFrame, { url: 'app://bundle/?' }) };
  state.listeners.get('xveon-unsaved')(event, { version: 2, edits: [{ id: 'a'.repeat(22), name: 'a', folder: { id: 'f', name: 'Photos' }, error: 'stopped' }] });
  state.quit.mockImplementationOnce(() => state.app.emit('before-quit', { preventDefault: vi.fn() }));
  state.workerEvent({ stopped: 'The background worker stopped.' });
  await vi.advanceTimersByTimeAsync(3001);
  expect(dialog.showMessageBox).toHaveBeenNthCalledWith(2, state.win, expect.objectContaining({ message: expect.stringContaining('1 photos in Photos'), buttons: ['Quit anyway', 'Cancel'] }));
  if (ending) state.win.emit('session-end');
  cancel({ response: 1, checkboxChecked: false }); await vi.advanceTimersByTimeAsync(3001);
  if (ending) expect(dialog.showMessageBox).toHaveBeenCalledTimes(2);
  else {
    expect(dialog.showMessageBox).toHaveBeenNthCalledWith(3, state.win, expect.objectContaining({ buttons: ['Restart', 'Quit'] }));
    expect(state.stop).not.toHaveBeenCalled();
  }
});
