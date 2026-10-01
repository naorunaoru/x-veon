import { contextBridge, ipcRenderer, webUtils } from 'electron';
import type { DesktopBridge } from '../protocol/bridge';
import { isBridgeEvent, isDesktopRequest, isUnsavedUpdate, isFlushResponse, type DesktopRequest } from '../protocol/security';
ipcRenderer.on('xveon-port', (event, data) => {
  if (data?.version !== 2 || event.ports.length !== 1) return;
  window.postMessage({ type: 'xveon-port', version: 2 }, location.origin, event.ports);
});
async function invoke(request: DesktopRequest) {
  if (!isDesktopRequest(request)) throw new Error('Invalid bridge request');
  const response: unknown = await ipcRenderer.invoke('xveon-desktop', request);
  if (request.kind === 'requestWorkerPort') return;
  const object = (value: unknown): value is Record<string, unknown> => value !== null && typeof value === 'object' && !Array.isArray(value);
  const dense = (value: unknown, check: (item: unknown) => boolean): boolean => Array.isArray(value)
    && Array.from({ length: value.length }, (_, i) => Object.hasOwn(value, i) && check(value[i])).every(Boolean);
  let valid = false;
  if (request.kind === 'recentFolders') valid = dense(response, item => object(item) && typeof item.id === 'string' && typeof item.name === 'string');
  else valid = response === null || (object(response) && typeof response.token === 'string'
    && (request.kind !== 'openDropped' || dense(response.selected, id => typeof id === 'string' && /^[A-Za-z0-9_-]{22}$/.test(id))));
  if (!valid || new TextEncoder().encode(JSON.stringify(response)).byteLength > 1_000_000) throw new Error('Invalid bridge response');
  return response;
}
const bridge: DesktopBridge = {
  version: 2,
  loadLast: () => invoke({ version: 2, kind: 'loadLast' }) as ReturnType<DesktopBridge['loadLast']>,
  openFolder: folderId => invoke({ version: 2, kind: 'openFolder', folderId }) as ReturnType<DesktopBridge['openFolder']>,
  openDropped: paths => invoke({ version: 2, kind: 'openDropped', paths }) as ReturnType<DesktopBridge['openDropped']>,
  recentFolders: () => invoke({ version: 2, kind: 'recentFolders' }) as ReturnType<DesktopBridge['recentFolders']>,
  pathsForFiles: files => files.map(file => webUtils.getPathForFile(file)),
  requestWorkerPort: () => invoke({ version: 2, kind: 'requestWorkerPort' }).then(() => {}),
  updateUnsaved(edits) {
    const message = { version: 2, edits };
    if (!isUnsavedUpdate(message)) throw new Error('Invalid unsaved inventory');
    ipcRenderer.send('xveon-unsaved', message);
  },
  respondFlush(requestId, unsaved) {
    const message = { version: 2, requestId, unsaved };
    if (!isFlushResponse(message)) throw new Error('Invalid flush response');
    ipcRenderer.send('xveon-flush', message);
  },
  onEvent(listener) {
    const receive = (_event: Electron.IpcRendererEvent, data: unknown) => { if (isBridgeEvent(data)) listener(data); };
    ipcRenderer.on('xveon-event', receive);
    return () => { ipcRenderer.removeListener('xveon-event', receive); };
  },
};
contextBridge.exposeInMainWorld('xveon', bridge);
