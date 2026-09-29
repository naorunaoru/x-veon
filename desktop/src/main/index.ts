import {
  app,
  powerMonitor,
  screen,
  BrowserWindow,
  ipcMain,
  MessageChannelMain,
  net,
  protocol,
  session,
  utilityProcess,
  type UtilityProcess,
} from 'electron';
import path from 'node:path';
import os from 'node:os';
import { readdir, mkdir, writeFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import { acceptsSender, assetName, isRequest } from '../protocol/security';
const root = path.resolve(__dirname, '../../..');
const evidence = path.join(root, 'tmp/m2-spike/runtime');
const runArg =
  process.argv.find((v) => v.startsWith('--spike-run='))?.split('=')[1] ??
  'manual';
if (!/^[a-z0-9-]+$/.test(runArg)) throw Error('Invalid spike run name');
app.setName('X-veon spike');
app.setPath('userData', path.join(root, 'tmp/m2-spike/user-data'));
protocol.registerSchemesAsPrivileged([
  {
    scheme: 'app',
    privileges: { standard: true, secure: true, supportFetchAPI: true },
  },
]);
let worker: UtilityProcess | null = null;
let win: BrowserWindow;
async function startWorker() {
  if (worker) return;
  worker = utilityProcess.fork(path.join(__dirname, 'worker.js'), [], {
    stdio: 'pipe',
    serviceName: 'X-veon spike worker',
  });
  worker.stdout?.on('data', (d) => console.log(String(d)));
  worker.stderr?.on('data', (d) => console.error(String(d)));
  worker.once('exit', () => {
    worker = null;
  });
}
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
const trusted = (event: Electron.IpcMainInvokeEvent) =>
  event.sender === win.webContents &&
  acceptsSender(
    event.senderFrame?.url ?? '',
    event.senderFrame === win.webContents.mainFrame,
  );
void app.whenReady().then(async () => {
  await mkdir(evidence, { recursive: true });
  await writeFile(
    path.join(evidence, `${runArg}-pid.json`),
    JSON.stringify({ pid: process.pid }),
  );
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
      "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; worker-src 'self' blob:; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-src 'none'",
    );
    return new Response(response.body, { status: response.status, headers });
  });
  session.defaultSession.setPermissionRequestHandler(
    (_wc, _permission, callback) => callback(false),
  );
  session.defaultSession.setPermissionCheckHandler(() => false);
  win = new BrowserWindow({
    width: 1440,
    height: 960,
    webPreferences: {
      sandbox: true,
      contextIsolation: true,
      nodeIntegration: false,
      preload: path.resolve(__dirname, '../preload/index.js'),
    },
  });
  win.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
  win.webContents.on('will-navigate', (e) => e.preventDefault());
  win.webContents.on('console-message', (event) =>
    console.log(`[renderer] ${event.message}`),
  );
  ipcMain.handle('xveon-request', async (event, request) => {
    if (!trusted(event) || !isRequest(request))
      throw Error('Invalid bridge request');
    if (request.kind === 'restart') {
      if (worker) {
        const old = worker;
        await new Promise<void>((resolve, reject) => {
          const timer = setTimeout(
            () => reject(Error('Worker exit timed out')),
            10_000,
          );
          old.once('exit', () => {
            clearTimeout(timer);
            resolve();
          });
          old.kill();
        });
      }
      await startWorker();
      return { restarted: true };
    }
    if (request.kind === 'diagnostics') {
      return {
        versions: process.versions,
        os: { release: os.release(), version: os.version() },
        onBatteryPower: powerMonitor.isOnBatteryPower(),
        display: screen.getDisplayMatching(win.getBounds()),
        launchArguments: process.argv,
        platform: process.platform,
        arch: process.arch,
        gpu: await app.getGPUInfo('complete'),
        metrics: app.getAppMetrics(),
        preferences: {
          sandbox: true,
          contextIsolation: true,
          nodeIntegration: false,
        },
      };
    }
    await startWorker();
    const { port1, port2 } = new MessageChannelMain();
    worker!.postMessage({ version: 1, kind: 'connect' }, [port1]);
    event.senderFrame!.postMessage('xveon-port', null, [port2]);
    return { connected: true };
  });
  ipcMain.handle('xveon-report', async (event, data) => {
    if (
      !trusted(event) ||
      data?.version !== 1 ||
      !['golden', 'timing', 'transport', 'capabilities', 'error'].includes(
        data.name,
      )
    )
      throw Error('Invalid report');
    const text = JSON.stringify(data.value, null, 2);
    if (text.length > 2_000_000) throw Error('Report too large');
    await writeFile(path.join(evidence, `${runArg}-${data.name}.json`), text);
    console.log(`[report] ${runArg}-${data.name}`);
  });
  const params = new URLSearchParams();
  const mode = process.argv
    .find((v) => v.startsWith('--spike='))
    ?.split('=')[1];
  if (mode === 'golden') params.set('golden', 'render');
  else if (mode) params.set('spike', mode);
  const sample = process.argv
    .find((v) => v.startsWith('--spike-sample='))
    ?.split('=')[1];
  if (sample) params.set('sample', sample);
  await win.loadURL('app://bundle/?' + params);
});
app.on('window-all-closed', () => app.quit());
app.on('before-quit', () => {
  worker?.kill();
});
