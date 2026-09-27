import { renderHook, fireEvent } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
const remove = vi.hoisted(() => vi.fn());
vi.mock('@/app/services/library', () => ({ removeFile: remove }));
import { useLibraryShortcuts } from './useLibraryShortcuts';
import { useAppStore } from '@/app/store';
beforeEach(() => { remove.mockClear(); useAppStore.setState({ selectedFileId: 'b' }); });
it.each(['Delete', 'Backspace'])('removes the selected photo with %s', (key) => {
  renderHook(useLibraryShortcuts);
  fireEvent.keyDown(document.body, { key });
  expect(remove).toHaveBeenCalledWith('b');
});
it('protects editing, dialogs, modifier shortcuts, and held keys', () => {
  renderHook(useLibraryShortcuts);
  for (const tag of ['input', 'textarea', 'select']) {
    const field = document.createElement(tag); document.body.append(field);
    fireEvent.keyDown(field, { key: 'Backspace' }); field.remove();
  }
  const editor = document.createElement('div'); editor.setAttribute('contenteditable', 'true'); document.body.append(editor);
  fireEvent.keyDown(editor, { key: 'Delete' }); editor.remove();
  fireEvent.keyDown(document.body, { key: 'Delete', repeat: true });
  fireEvent.keyDown(document.body, { key: 'Backspace', metaKey: true });
  const dialog = document.createElement('div'); dialog.setAttribute('role', 'dialog'); document.body.append(dialog);
  fireEvent.keyDown(document.body, { key: 'Delete' }); dialog.remove();
  expect(remove).not.toHaveBeenCalled();
});
