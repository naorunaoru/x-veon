import { useRef, useState } from 'react';
import { getHost } from '@/app/services/host';
import { clearLibrary, importFiles } from '@/app/services/library';
import { RAW_ACCEPT } from '@/lib/catalog';
import { Dialog, DialogContent } from '@/components/dialogs/Dialog';
import { useMenu } from './useMenu';

export function LibraryMenu() {
  const library = getHost().library;
  const menu = useMenu();
  const inputRef = useRef<HTMLInputElement>(null);
  const [confirm, setConfirm] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (library.openFolder) return null;

  async function clear() {
    setClearing(true);
    setError(null);
    try {
      await clearLibrary();
      setConfirm(false);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setClearing(false);
    }
  }

  return (
    <div className="xv-folder-menu" ref={menu.rootRef}>
      <button type="button" className="xv-folder-menu__trigger xv-glass" {...menu.triggerProps}>
        Library
      </button>
      {error && !confirm && <p role="alert" className="xv-folder-menu__notice">{error}</p>}
      {menu.open && (
        <div className="xv-folder-menu__list xv-glass-heavy" {...menu.menuProps}>
          <button
            type="button"
            role="menuitem"
            className="xv-folder-menu__item"
            onClick={() => {
              menu.close();
              inputRef.current?.click();
            }}
          >
            Add files…
          </button>
          {library.clear && (
            <button
              type="button"
              role="menuitem"
              className="xv-folder-menu__item"
              onClick={() => {
                menu.close();
                setConfirm(true);
                setError(null);
              }}
            >
              Clear library
            </button>
          )}
        </div>
      )}
      <input
        ref={inputRef}
        type="file"
        accept={RAW_ACCEPT}
        multiple
        hidden
        onChange={(e) => {
          if (e.target.files) void importFiles(Array.from(e.target.files));
          e.target.value = '';
        }}
      />
      <Dialog
        open={confirm}
        onOpenChange={(open) => {
          if (!clearing) setConfirm(open);
        }}
      >
        <DialogContent
          title="Clear library"
          actions={
            <>
              <button className="xv-btn" disabled={clearing} onClick={() => setConfirm(false)}>
                Cancel
              </button>
              <button className="xv-btn" disabled={clearing} onClick={clear}>
                Remove photos, edits and settings
              </button>
            </>
          }
        >
          <p>Removes every photo, edit and setting this build has stored in this browser.</p>
          {error && <p role="alert">{error}</p>}
        </DialogContent>
      </Dialog>
    </div>
  );
}
