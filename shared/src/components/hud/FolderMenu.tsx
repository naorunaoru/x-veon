import { useEffect, useState } from 'react';
import { useAppStore } from '@/app/store';
import { getHost } from '@/app/services/host';
import { openFolder } from '@/app/services/library';
import { useMenu } from './useMenu';

type FolderRef = NonNullable<ReturnType<typeof useAppStore.getState>['folder']>;

export function FolderMenu() {
  const folder = useAppStore(state => state.folder);
  const library = getHost().library;
  const menu = useMenu();
  const [recent, setRecent] = useState<FolderRef[]>([]);

  useEffect(() => {
    if (!menu.open) return;
    let active = true;
    void library.recentFolders?.().then(folders => {
      if (active) setRecent(folders);
    }).catch(error => console.warn('Recent folders failed:', error));
    return () => { active = false; };
  }, [menu.open, library]);

  if (!library.openFolder) return null;

  const choose = (selected?: FolderRef) => {
    menu.close();
    void openFolder(selected).catch(error => console.warn('Folder open failed:', error));
  };

  return (
    <div className="xv-folder-menu" ref={menu.rootRef}>
      <button type="button" className="xv-folder-menu__trigger xv-glass" {...menu.triggerProps}>
        {folder?.name ?? 'Open folder…'}
      </button>
      {menu.open && (
        <div className="xv-folder-menu__list xv-glass-heavy" {...menu.menuProps}>
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
