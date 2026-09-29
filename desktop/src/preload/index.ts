import { contextBridge, ipcRenderer } from 'electron';
import type { SpikeBridge } from '../protocol/bridge';
ipcRenderer.on('xveon-port', (event) => {
  window.postMessage(
    { type: 'xveon-port', version: 1 },
    location.origin,
    event.ports,
  );
});
const bridge: SpikeBridge = {
  version: 1,
  environment: {
    sandboxed: process.sandboxed,
    contextIsolated: process.contextIsolated,
  },
  request: (kind) => ipcRenderer.invoke('xveon-request', { version: 1, kind }),
  report: (name, value) =>
    ipcRenderer.invoke('xveon-report', { version: 1, name, value }),
};
contextBridge.exposeInMainWorld('xveon', bridge);
