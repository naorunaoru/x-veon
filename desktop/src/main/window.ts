import { BrowserWindow, session, shell } from 'electron';
export function createMainWindow(opts: { preload: string }): BrowserWindow {
  session.defaultSession.setPermissionRequestHandler((_wc, _permission, callback) => callback(false));
  session.defaultSession.setPermissionCheckHandler(() => false);
  const win = new BrowserWindow({ width: 1440, height: 960, webPreferences: { sandbox: true, contextIsolation: true, nodeIntegration: false, backgroundThrottling: false, ...opts } });
  win.webContents.on('will-navigate', event => event.preventDefault());
  win.webContents.setWindowOpenHandler(({ url }) => {
    try { if (['http:', 'https:'].includes(new URL(url).protocol)) void shell.openExternal(url).catch(() => {}); } catch { /* Invalid URLs stay blocked. */ }
    return { action: 'deny' };
  });
  return win;
}
