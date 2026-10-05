import { contextBridge, ipcRenderer, webUtils } from 'electron';
import type { DesktopBridge } from '../protocol/bridge';
import { isBridgeEvent, isDesktopRequest, isUnsavedUpdate, isFlushResponse, isWorkerPortDelivery, type DesktopRequest } from '../protocol/security';
let requestedPort: string | undefined;
ipcRenderer.on('xveon-port', (event, data: unknown) => {
  if (!isWorkerPortDelivery(data) || data.requestId !== requestedPort || event.ports.length !== 1) {
    for (const port of event.ports) port.close();
    return;
  }
  requestedPort = undefined;
  window.postMessage({ type: 'xveon-port', version: 2, requestId: data.requestId }, location.origin, event.ports);
});
async function invoke(request: DesktopRequest) {
  if (!isDesktopRequest(request)) throw new Error('Invalid bridge request');
  const response: unknown = await ipcRenderer.invoke('xveon-desktop', request);
  if (request.kind === 'requestWorkerPort' || request.kind === 'revealExport') return;
  const object = (value: unknown): value is Record<string, unknown> => value !== null && typeof value === 'object' && !Array.isArray(value);
  const dense = (value: unknown, check: (item: unknown) => boolean): boolean => Array.isArray(value)
    && Array.from({ length: value.length }, (_, i) => Object.hasOwn(value, i) && check(value[i])).every(Boolean);
  let valid = false;
  if (request.kind === 'chooseExportDestination') valid = response === null || (object(response) && typeof response.token === 'string' && /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(response.token) && typeof response.name === 'string' && response.name.length > 0 && response.name.length <= 255);
  else if (request.kind === 'recentFolders') valid = dense(response, item => object(item) && typeof item.id === 'string' && typeof item.name === 'string');
  else valid = response === null || (object(response) && typeof response.token === 'string'
    && (request.kind !== 'openDropped' || dense(response.selected, id => typeof id === 'string' && /^[A-Za-z0-9_-]{22}$/.test(id))));
  if (!valid || new TextEncoder().encode(JSON.stringify(response)).byteLength > 1_000_000) throw new Error('Invalid bridge response');
  return response;
}
const bridge: DesktopBridge = {
  version: 2,
  chooseExportDestination: (photoId, format) => invoke({ version: 2, kind: 'chooseExportDestination', photoId, format }) as ReturnType<DesktopBridge['chooseExportDestination']>,
  revealExport: async token => { await invoke({ version: 2, kind: 'revealExport', token }); },
  loadLast: () => invoke({ version: 2, kind: 'loadLast' }) as ReturnType<DesktopBridge['loadLast']>,
  openFolder: folderId => invoke({ version: 2, kind: 'openFolder', folderId }) as ReturnType<DesktopBridge['openFolder']>,
  async openDropped(files) {
    if (!Array.isArray(files) || files.some(file => !(file instanceof File))) throw new Error('Only files on disk can be opened.');
    const paths = Array.from(files, file => webUtils.getPathForFile(file));
    if (paths.some(path => !path)) throw new Error('Only files on disk can be opened.');
    return invoke({ version: 2, kind: 'openDropped', paths }) as ReturnType<DesktopBridge['openDropped']>;
  },
  recentFolders: () => invoke({ version: 2, kind: 'recentFolders' }) as ReturnType<DesktopBridge['recentFolders']>,
  async requestWorkerPort(requestId) {
    const request = { version: 2 as const, kind: 'requestWorkerPort' as const, requestId };
    if (!isDesktopRequest(request)) throw new Error('Invalid bridge request');
    requestedPort = requestId;
    try { await invoke(request); }
    catch (error) { if (requestedPort === requestId) requestedPort = undefined; throw error; }
  },
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
