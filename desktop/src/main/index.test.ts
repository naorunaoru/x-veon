import { afterEach, expect, it, vi } from 'vitest';
import os from 'node:os';
import fs from 'node:fs/promises';
import path from 'node:path';
const state = vi.hoisted(() => ({ mkdir: vi.fn(async () => {}), displayRead: vi.fn(() => ({ potentialEdr: 16 })), displayLoad: vi.fn(), destinations: vi.fn(), folderFile: undefined as string | undefined, boot: undefined as Promise<unknown> | undefined, handles: new Map<string, any>(), listeners: new Map<string, any>(), privileges: vi.fn(), name: vi.fn(), quit: vi.fn(), stop: vi.fn(), win: undefined as any, windowOptions: undefined as any, app: undefined as any, report: vi.fn(async () => {}), workerEvent: undefined as any, restart: vi.fn(async () => {}), lock: vi.fn(() => true) }));
vi.mock('electron', async () => {
  const { EventEmitter } = await import('node:events');
  const app = Object.assign(new EventEmitter(), { setName: state.name, requestSingleInstanceLock: state.lock, setPath: vi.fn(), getPath: () => '/tmp/task9-userdata', whenReady: () => ({ then: (fn: () => Promise<unknown>) => { state.boot = fn(); return state.boot; } }), quit: state.quit, exit: vi.fn() }); state.app = app;
  const win = Object.assign(new EventEmitter(), { webContents: Object.assign(new EventEmitter(), { send: vi.fn(), setWindowOpenHandler: vi.fn(), mainFrame: {}, executeJavaScript: vi.fn() }), loadURL: vi.fn(async () => {}), destroy: vi.fn(), isDestroyed: () => false, getNativeWindowHandle: () => Buffer.from([1, 2, 3, 4, 5, 6, 7, 8]), isMinimized: () => true, restore: vi.fn(), focus: vi.fn() }); state.win = win;
  return { app, BrowserWindow: class { constructor(options: unknown) { state.windowOptions = options; return win; } }, powerMonitor: new EventEmitter(), screen: {}, ipcMain: { handle: (name: string, fn: any) => state.handles.set(name, fn), on: (name: string, fn: any) => state.listeners.set(name, fn) }, MessageChannelMain: class {}, net: {}, protocol: { registerSchemesAsPrivileged: state.privileges, handle: vi.fn() }, session: { defaultSession: { setPermissionRequestHandler: vi.fn(), setPermissionCheckHandler: vi.fn() } }, utilityProcess: {}, Menu: { buildFromTemplate: (t: unknown) => t, setApplicationMenu: vi.fn() }, dialog: { showSaveDialog: vi.fn(async () => ({ canceled: true })), showOpenDialog: vi.fn(async () => ({ canceled: true, filePaths: [] })), showErrorBox: vi.fn(), showMessageBox: vi.fn(async () => ({ response: 1 })) }, shell: { showItemInFolder: vi.fn() } };
});
vi.mock('node:fs/promises', async importOriginal => ({ ...await importOriginal<typeof import('node:fs/promises')>(), readdir: vi.fn(async () => []), mkdir: state.mkdir, writeFile: vi.fn(async () => {}) }));
vi.mock('./folders', async importOriginal => {
 const actual = await importOriginal<typeof import('./folders')>();
 return { ...actual, createFolderStore: () => state.folderFile ? actual.createFolderStore(state.folderFile) : ({ recent: () => [], forgetPath: () => false, last: () => null, resolve: () => null }) };
});
vi.mock('./worker', () => ({ createWorkerSupervisor: (opts: any) => { state.workerEvent = opts.onEvent; return ({ restart: state.restart, roots: [], registry: new Map(), onMessage() {}, stop: state.stop, request: vi.fn(), connect: vi.fn() }); } }));
vi.mock('./golden-report', () => ({ watchGoldenReport: state.report }));
vi.mock('./exports', async importOriginal => { const actual = await importOriginal<typeof import('./exports')>(); return { createExportDestinations: (deps: Parameters<typeof actual.createExportDestinations>[0]) => { state.destinations(deps); return actual.createExportDestinations(deps); } }; });
vi.mock('./display', () => ({ loadDisplayReader: () => { state.displayLoad(); return { read: state.displayRead }; } }));
const originalArgs = [...process.argv];
afterEach(async () => { const { dialog, powerMonitor } = await import('electron'); state.app?.removeAllListeners(); state.win?.removeAllListeners(); state.win?.webContents.removeAllListeners(); powerMonitor.removeAllListeners(); state.quit.mockReset(); vi.mocked(dialog.showMessageBox).mockReset().mockResolvedValue({ response: 1, checkboxChecked: false }); process.argv = [...originalArgs]; vi.useRealTimers(); vi.unstubAllGlobals(); vi.clearAllMocks(); state.handles.clear(); state.listeners.clear(); });
it('reads the main window display with its native handle', async () => {
  vi.resetModules(); await import('./index'); await state.boot;
  const event = { sender: state.win.webContents, senderFrame: Object.assign(state.win.webContents.mainFrame, { url: 'app://bundle/?' }) };
  await expect(state.handles.get('xveon-desktop')(event, { version: 2, kind: 'displayReadings' })).resolves.toEqual({ potentialEdr: 16 });
  expect(state.displayRead).toHaveBeenCalledWith(Buffer.from([1, 2, 3, 4, 5, 6, 7, 8]));
});
it('boots with only the desktop route, registers a secure streaming scheme, and defers worker cleanup until quit is accepted', async () => {
  vi.resetModules();
  process.argv.push('--golden-report=/ignored.json');
  vi.useFakeTimers(); vi.stubGlobal('__XV_GOLDEN__', false); await import('./index'); await state.boot;
  expect([...state.handles.keys()]).toEqual(['xveon-desktop']); expect(state.handles.has('xveon-desktop')).toBe(true);
  expect(state.privileges.mock.calls[0][0]).toContainEqual({ scheme: 'xveon-photo', privileges: { standard: true, secure: true, supportFetchAPI: true, corsEnabled: true, stream: true } });
  expect(state.name).toHaveBeenCalledWith('X-veon Dev'); expect(state.win.loadURL).toHaveBeenCalledWith('app://bundle/?');
  expect(state.windowOptions.webPreferences.backgroundThrottling).toBe(false);
  const preventDefault = vi.fn(); state.app.emit('before-quit', { preventDefault }); expect(preventDefault).toHaveBeenCalledOnce(); expect(state.stop).not.toHaveBeenCalled();
  let finish!: () => void;
  state.stop.mockImplementationOnce(() => new Promise<void>(resolve => { finish = resolve; }));
  await vi.advanceTimersByTimeAsync(3001);
  expect(state.stop).toHaveBeenCalledOnce(); expect(state.quit).not.toHaveBeenCalled();
  state.app.emit('before-quit', { preventDefault }); expect(preventDefault).toHaveBeenCalledTimes(2);
  finish(); await vi.advanceTimersByTimeAsync(0); expect(state.quit).toHaveBeenCalledOnce();
});

it.each([undefined, 'full', 'render', 'bench', 'unknown'])('loads golden mode %s and fixes its export destination', async mode => {
  vi.resetModules(); vi.stubGlobal('__XV_GOLDEN__', true); process.argv.push('--golden-report=/report.json'); if (mode) process.argv.push(`--golden-mode=${mode}`);
  await import('./index'); await state.boot;
  expect(state.win.loadURL).toHaveBeenCalledWith(`app://bundle/?golden=${mode === 'render' || mode === 'bench' ? mode : 'full'}`);
  expect(state.mkdir).toHaveBeenCalledWith('/report.json.exports', { recursive: true });
  expect(state.destinations).toHaveBeenCalledWith(expect.objectContaining({ fixedDir: '/report.json.exports' })); expect(state.report).toHaveBeenCalledWith(state.win, '/report.json');
  expect(state.windowOptions.webPreferences.backgroundThrottling).toBe(false);
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
  expect(dialog.showMessageBox).toHaveBeenNthCalledWith(2, state.win, expect.objectContaining({ message: 'Unsaved edits', detail: "1 photo has edits that aren't saved.\n\na — Photos", buttons: ['Quit anyway', 'Cancel'] }));
  if (ending) state.win.emit('session-end');
  cancel({ response: 1, checkboxChecked: false }); await vi.advanceTimersByTimeAsync(3001);
  if (ending) expect(dialog.showMessageBox).toHaveBeenCalledTimes(2);
  else {
    expect(dialog.showMessageBox).toHaveBeenNthCalledWith(3, state.win, expect.objectContaining({ buttons: ['Restart', 'Quit'] }));
    expect(state.stop).not.toHaveBeenCalled();
  }
});

it('stops a second instance before starting services and focuses the owned window', async () => {
 vi.resetModules(); await import('./index'); await state.boot;
 state.app.emit('second-instance'); expect(state.win.restore).toHaveBeenCalledOnce(); expect(state.win.focus).toHaveBeenCalledOnce();
 vi.resetModules(); state.lock.mockReturnValueOnce(false); state.boot = undefined;
 await import('./index'); expect(state.quit).toHaveBeenCalled(); expect(state.boot).toBeUndefined();
});

it('sets the explicit isolated profile before acquiring its instance lock', async () => {
 const profile = path.join(os.tmpdir(), 'xveon-isolated-profile');
 vi.resetModules(); process.argv.push(`--user-data-dir=${profile}`); await import('./index'); await state.boot;
 expect(state.app.setPath).toHaveBeenCalledWith('userData', profile);
 expect(state.app.setPath.mock.invocationCallOrder.at(-1)).toBeLessThan(state.lock.mock.invocationCallOrder.at(-1)!);
});

it('reports folder failures using an asynchronous error dialog owned by the main window', async () => {
 vi.resetModules(); await import('./index'); await state.boot;
 const { dialog } = await import('electron');
 vi.mocked(dialog.showOpenDialog).mockRejectedValueOnce(new Error('EACCES: unavailable folder'));
 vi.mocked(dialog.showMessageBox).mockImplementationOnce(() => new Promise(() => {}));
 const event = { sender: state.win.webContents, senderFrame: Object.assign(state.win.webContents.mainFrame, { url: 'app://bundle/?' }) };
 await expect(state.handles.get('xveon-desktop')(event, { version: 2, kind: 'openFolder' })).resolves.toBeNull();
 expect(dialog.showMessageBox).toHaveBeenCalledWith(state.win, expect.objectContaining({ type: 'error', detail: 'EACCES: unavailable folder' }));
 expect(dialog.showErrorBox).not.toHaveBeenCalled();
});

it('boots and serves saved recent folders without probing their potentially offline paths', async () => {
 const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'startup-recents-'));
 const recent = Array.from({ length: 10 }, (_, i) => ({ id: `recent-${i}`, name: `Offline ${i}`, path: path.join(dir, `offline-${i}`), openedAt: '2026-10-05' }));
 const paths = new Set(recent.map(entry => entry.path));
 state.folderFile = path.join(dir, 'folders.json');
 await fs.writeFile(state.folderFile, JSON.stringify({ version: 1, last: recent[0].id, recent }));
 const originalStat = fs.stat.bind(fs), originalRealpath = fs.realpath.bind(fs);
 const stat = vi.spyOn(fs, 'stat').mockImplementation(target => paths.has(String(target)) ? new Promise(() => {}) : originalStat(target));
 const realpath = vi.spyOn(fs, 'realpath').mockImplementation(target => paths.has(String(target)) ? new Promise(() => {}) : originalRealpath(target));
 try {
   vi.resetModules(); await import('./index'); await state.boot;
   const event = { sender: state.win.webContents, senderFrame: Object.assign(state.win.webContents.mainFrame, { url: 'app://bundle/?' }) };
   const expected = recent.map(({ id, name }) => ({ id, name }));
   await expect(state.handles.get('xveon-desktop')(event, { version: 2, kind: 'recentFolders' })).resolves.toEqual(expected);
   expect(state.win.loadURL).toHaveBeenCalledWith('app://bundle/?');
   expect([...stat.mock.calls, ...realpath.mock.calls].filter(([target]) => paths.has(String(target)))).toEqual([]);
 } finally { state.folderFile = undefined; stat.mockRestore(); realpath.mockRestore(); await fs.rm(dir, { recursive: true, force: true }); }
});
