import type { MenuItemConstructorOptions } from 'electron';
import type { FolderRef } from '@/host';
export type MenuAction = { kind: 'folder-request'; folderId?: string };
export function buildMenuTemplate(recent: FolderRef[], send: (action: MenuAction) => void, platform: NodeJS.Platform): MenuItemConstructorOptions[] {
  return [
    ...(platform === 'darwin' ? [{ role: 'appMenu' as const }] : []),
    { label: 'File', submenu: [
      { label: 'Open Folder…', accelerator: 'CmdOrCtrl+O', click: () => send({ kind: 'folder-request' }) },
      { label: 'Open Recent', enabled: recent.length > 0, submenu: recent.map(folder => ({ label: folder.name, click: () => send({ kind: 'folder-request', folderId: folder.id }) })) },
      { type: 'separator' }, { role: 'close' }, { role: 'quit' },
    ] },
    { role: 'editMenu' }, { role: 'windowMenu' },
  ];
}
