import { getHost } from '@/app/services/host';
import { useEffect } from 'react';
import { useAppStore } from '@/app/store';
import { removeFile } from '@/app/services/library';

/** Keys typed here belong to the field, dialog or menu, not the library. */
const OWNS_KEYS = 'input, textarea, select, [contenteditable]:not([contenteditable="false"]), [role="slider"], [role="textbox"], [role="dialog"], [role="alertdialog"], [role="menu"]';
/** While one of these is open, the library keys are off. */
const OVERLAY = '[role="dialog"], [role="alertdialog"], [role="menu"]';

export function useLibraryShortcuts() {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented || event.isComposing) return;
      const target = event.target;
      if (target instanceof Element && target.closest(OWNS_KEYS)) return;
      if (document.querySelector(OVERLAY)) return;

      const state = useAppStore.getState();
      const command = event.metaKey || event.ctrlKey;
      const plain = !command && !event.altKey && !event.shiftKey;
      const rowStep = state.columnWidth >= 2 ? state.columnWidth : 1;
      const step =
        event.key === 'ArrowLeft' ? -1 : event.key === 'ArrowRight' ? 1
        : event.key === 'ArrowUp' ? -rowStep : event.key === 'ArrowDown' ? rowStep
        : 0;

      if (step !== 0 && !command && !event.altKey) {
        event.preventDefault();
        state.moveCurrent(step, event.shiftKey);
      } else if (event.key === 'a' && command && !event.altKey && !event.shiftKey) {
        event.preventDefault();
        state.selectAllShown();
      } else if (event.key === 'Escape' && plain) {
        event.preventDefault();
        state.clearSelection();
      } else if (event.key === 'Tab' && plain && !event.repeat) {
        event.preventDefault();
        state.toggleColumn();
      } else if ((event.key === 'Delete' || event.key === 'Backspace') && plain && !event.repeat) {
        if (!getHost().library.remove || !state.currentId) return;
        event.preventDefault();
        removeFile(state.currentId);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);
}
