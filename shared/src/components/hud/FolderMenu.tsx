import { useEffect, useRef, useState } from 'react';
import type { KeyboardEvent } from 'react';
import { useAppStore } from '@/app/store';
import { getHost } from '@/app/services/host';
import { openFolder } from '@/app/services/library';

type FolderRef = NonNullable<ReturnType<typeof useAppStore.getState>['folder']>;

export function FolderMenu() {
  const folder = useAppStore(state => state.folder);
  const library = getHost().library;
  const [open, setOpen] = useState(false);
  const [recent, setRecent] = useState<FolderRef[]>([]);
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    let active = true;
    void library.recentFolders?.().then(folders => {
      if (active) setRecent(folders);
    }).catch(error => console.warn('Recent folders failed:', error));
    const onPointerDown = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener('pointerdown', onPointerDown);
    return () => {
      active = false;
      document.removeEventListener('pointerdown', onPointerDown);
    };
  }, [open, library]);

  useEffect(() => {
    if (open) menuRef.current?.querySelector<HTMLButtonElement>('[role="menuitem"]')?.focus();
  }, [open, recent]);

  if (!library.openFolder) return null;

  const choose = (selected?: FolderRef) => {
    setOpen(false);
    triggerRef.current?.focus();
    void openFolder(selected).catch(error => console.warn('Folder open failed:', error));
  };

  const onMenuKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const items = Array.from(menuRef.current?.querySelectorAll<HTMLButtonElement>('[role="menuitem"]') ?? []);
    const current = items.indexOf(document.activeElement as HTMLButtonElement);
    if (event.key === 'Escape') {
      event.preventDefault();
      setOpen(false);
      triggerRef.current?.focus();
    } else if (event.key === 'ArrowDown' || event.key === 'ArrowUp' || event.key === 'Home' || event.key === 'End') {
      event.preventDefault();
      const next = event.key === 'Home' ? 0 : event.key === 'End' ? items.length - 1
        : (current + (event.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length;
      items[next]?.focus();
    }
  };

  return (
    <div className="xv-folder-menu" ref={rootRef}>
      <button
        type="button"
        className="xv-folder-menu__trigger xv-glass"
        ref={triggerRef}
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen(value => !value)}
        onKeyDown={event => {
          if (event.key === 'ArrowDown') {
            event.preventDefault();
            setOpen(true);
          }
        }}
      >
        {folder?.name ?? 'Open folder…'}
      </button>
      {open && (
        <div className="xv-folder-menu__list xv-glass-heavy" role="menu" ref={menuRef} onKeyDown={onMenuKeyDown}>
          {recent.map(item => (
            <button key={item.id} type="button" role="menuitem" className="xv-folder-menu__item" onClick={() => choose(item)}>
              {item.name}
            </button>
          ))}
          <button type="button" role="menuitem" className="xv-folder-menu__item" onClick={() => choose()}>
            Open folder…
          </button>
        </div>
      )}
    </div>
  );
}
