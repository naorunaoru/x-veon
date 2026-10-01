import {
  app,
  powerMonitor,
  BrowserWindow,
  ipcMain,
  net,
  protocol,
  dialog,
  Menu,
  utilityProcess,
} from 'electron';
import path from 'node:path';
import { randomBytes } from 'node:crypto';
import { BUILD, channelLabel } from '@/lib/channel';
import { createFolderStore } from './folders';
import { createFolderRequests } from './folder-requests';
import { createWorkerSupervisor } from './worker';
import { registerPhotoProtocol } from './photo-protocol';
import { buildMenuTemplate } from './menu';
import { createMainWindow } from './window';
import { createCloseGuard, unsavedQuitMessage } from './close-guard';
import { registerDesktopIpc } from './desktop-ipc';
import type { BridgeEvent } from '../protocol/bridge';
import { readdir, mkdir } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import { acceptsSender, assetName, isBridgeEvent, CONTENT_SECURITY_POLICY } from '../protocol/security';
app.setName(BUILD.channel === 'stable' ? 'X-veon' : `X-veon ${channelLabel(BUILD.channel)}`);
protocol.registerSchemesAsPrivileged([
  { scheme: 'xveon-photo', privileges: { standard: true, secure: true, supportFetchAPI: true, corsEnabled: true, stream: true } },
  {
    scheme: 'app',
    privileges: { standard: true, secure: true, supportFetchAPI: true },
  },
]);
let win: BrowserWindow;
async function assets(dir: string, prefix = ''): Promise<string[]> {
  const result: string[] = [];
  for (const file of await readdir(dir, { withFileTypes: true })) {
    if (file.isSymbolicLink()) continue;
    const name = prefix + file.name;
    if (file.isDirectory())
      result.push(...(await assets(path.join(dir, file.name), name + '/')));
    else result.push(name);
  }
  return result;
}
const trusted = (event: Electron.IpcMainInvokeEvent | Electron.IpcMainEvent) =>
  event.sender === win.webContents &&
  acceptsSender(
    event.senderFrame?.url ?? '',
    event.senderFrame === win.webContents.mainFrame,
  );
void app.whenReady().then(async () => {
  const bundle = path.resolve(__dirname, '../renderer'),
    files = new Set(await assets(bundle));
  protocol.handle('app', async (request) => {
    const name = assetName(request.url, request.method, files);
    if (!name) return new Response('Not found', { status: 404 });
    const response = await net.fetch(
      pathToFileURL(path.join(bundle, name)).toString(),
    );
    const headers = new Headers(response.headers);
    if (name.endsWith('.wasm')) headers.set('Content-Type', 'application/wasm');
    headers.set(
      'Content-Security-Policy',
      CONTENT_SECURITY_POLICY,
    );
    return new Response(response.body, { status: response.status, headers });
  });
  const goldenFile = typeof __XV_GOLDEN__ !== 'undefined' && __XV_GOLDEN__
    ? process.argv.find(arg => arg.startsWith('--golden-report='))?.slice('--golden-report='.length) : undefined;
  win = createMainWindow({ preload: path.resolve(__dirname, '../preload/index.js'), backgroundThrottling: !goldenFile });
  const send = (event: BridgeEvent) => {
    const message = { version: 2, ...event };
    if (!isBridgeEvent(message)) throw new Error('Invalid bridge event');
    if (!win.isDestroyed()) win.webContents.send('xveon-event', message);
  };
  const cacheDir = path.join(app.getPath('userData'), 'cache');
  await mkdir(cacheDir, { recursive: true });
  const store = createFolderStore(path.join(app.getPath('userData'), 'folders.json'));
  let stoppedReason: string | null = null;
  let recoveryOpen = false, endingSession = false;
  async function recoverWorker(): Promise<void> {
    if (!stoppedReason || recoveryOpen || endingSession) return;
    recoveryOpen = true;
    let response: number;
    try {
      ({ response } = await dialog.showMessageBox(win, { type: 'error', message: 'Background worker stopped', detail: stoppedReason,
        buttons: ['Restart', 'Quit'], defaultId: 0, cancelId: 1, noLink: true,
      }));
    } finally { recoveryOpen = false; }
    if (endingSession) return;
    if (response === 0) await supervisor.restart();
    else app.quit(); // before-quit goes through the unsaved-edit guard below.
  }
  const supervisor = createWorkerSupervisor({
    fork: () => utilityProcess.fork(path.join(__dirname, 'worker.js'), [], { stdio: 'pipe', serviceName: 'X-veon library' }),
    sessionKey: randomBytes(32), cacheDir,
    onEvent: event => {
      if (event === 'restarted') { stoppedReason = null; send({ kind: 'worker-restarted', worker: supervisor.instance }); }
      else {
        stoppedReason = event.stopped;
        send({ kind: 'worker-stopped', reason: event.stopped });
        void recoverWorker().catch(() => {}); // A failed restart reports another stopped event.
      }
    },
  });
  const refreshMenu = () => Menu.setApplicationMenu(Menu.buildFromTemplate(buildMenuTemplate(store.recent(), send, process.platform)));
  const folders = createFolderRequests({ store, worker: supervisor, send: frame => send({ kind: 'listing', frame }), accepted: refreshMenu,
    chooseFolder: async () => { const result = await dialog.showOpenDialog(win, { properties: ['openDirectory'] }); return result.canceled ? null : result.filePaths[0] ?? null; },
    error: (folder, message) => dialog.showErrorBox(`Could not open ${folder}`, message),
  });
  const ipc = registerDesktopIpc({ ipc: ipcMain, trusted, folders, recent: store.recent, connect: supervisor.connect, send });
  registerPhotoProtocol({ protocol, registry: supervisor.registry, roots: supervisor.roots, cacheDir,
    thumbnail: async id => (await supervisor.request({ kind: 'thumbnail', id })).path,
  });
  const stopWorkers = () => { endingSession = true; supervisor.stop(); };
  const guard = createCloseGuard({ window: win, app: { on: (event, handler) => { app.on(event, handler); }, quit: () => app.quit(), exit: code => { stopWorkers(); app.exit(code); } },
    inventory: ipc.inventory, requestFlush: ipc.requestFlush,
    confirmQuit: async unsaved => {
      const quit = (await dialog.showMessageBox(win, { type: 'warning', message: unsavedQuitMessage(unsaved), buttons: ['Quit anyway', 'Cancel'], defaultId: 1, cancelId: 1, noLink: true })).response === 0;
      if (!quit) void recoverWorker().catch(() => {});
      return quit;
    },
  });
  powerMonitor.on('shutdown', (event?: Electron.Event) => { event?.preventDefault(); endingSession = true; guard.onSessionEnd(); });
  win.on('session-end', () => { endingSession = true; guard.onSessionEnd(); });
  app.on('will-quit', stopWorkers);
  refreshMenu();
  win.webContents.on('console-message', (event) =>
    console.log(`[renderer] ${event.message}`),
  );
  if (goldenFile) {
    await win.loadURL('app://bundle/?golden=render');
    const { watchGoldenReport } = await import('./golden-report');
    await watchGoldenReport(win, goldenFile);
  } else await win.loadURL('app://bundle/?');
});
