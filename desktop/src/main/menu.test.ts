import { expect, it } from 'vitest';
import { buildMenuTemplate } from './menu';
it('keeps recent order and emits folder requests from open commands', () => {
  const events: unknown[] = [], template = buildMenuTemplate([{ id: 'b', name: 'Beach' }, { id: 'a', name: 'Alps' }], e => events.push(e), 'darwin');
  const file = template.find(m => m.label === 'File')!; const items = file.submenu as any[];
  expect(items[0].accelerator).toBe('CmdOrCtrl+O'); items[0].click();
  const recent = items.find(m => m.label === 'Open Recent').submenu;
  expect(recent.map((m: any) => m.label)).toEqual(['Beach', 'Alps']); recent[1].click();
  expect(events).toEqual([{ kind: 'folder-request' }, { kind: 'folder-request', folderId: 'a' }]);
  expect(items.some(m => m.role === 'quit')).toBe(true); expect(template.some(m => m.role === 'editMenu')).toBe(true); expect(template.some(m => m.role === 'windowMenu')).toBe(true);
});

it.each(['darwin', 'win32', 'linux'] as const)('provides the native close command on %s', platform => {
  const template = buildMenuTemplate([], () => {}, platform);
  const items = template.find(menu => menu.label === 'File')!.submenu as any[];
  const close = items.find(item => item.role === 'close');
  expect(close).toBeDefined();
  // Electron supplies the standard accelerator and closes the focused window,
  // which reaches the existing unsaved-edit guard.
  expect(close.accelerator).toBeUndefined();
  expect(close.click).toBeUndefined();
});
