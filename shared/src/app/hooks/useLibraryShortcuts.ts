import { getHost } from '@/app/services/host';
import { useEffect } from 'react';
import { useAppStore } from '@/app/store';
import { removeFile } from '@/app/services/library';

export function useLibraryShortcuts() {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (!getHost().library.remove) return;
      if (event.defaultPrevented || event.repeat || event.isComposing || event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) return;
      if (event.key !== 'Delete' && event.key !== 'Backspace') return;
      const target = event.target;
      if (target instanceof Element && target.closest('input, textarea, select, [contenteditable]:not([contenteditable="false"]), [role="slider"], [role="textbox"], [role="dialog"], [role="alertdialog"]')) return;
      if (document.querySelector('[role="dialog"], [role="alertdialog"]')) return;
      const id = useAppStore.getState().selectedFileId;
      if (!id) return;
      event.preventDefault();
      removeFile(id);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);
}
