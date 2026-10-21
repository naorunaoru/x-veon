import { FileMetaPill } from './FileMetaPill';
import { ZoomPill } from './ZoomPill';
import { FolderMenu } from './FolderMenu';
import { LibraryMenu } from './LibraryMenu';
import { getHost } from '@/app/services/host';
import './TopBar.css';

export function TopBar({ folderOnly = false }: { folderOnly?: boolean }) {
  return (
    <div className="xv-topbar">
      <div className="xv-topbar__left">
        {getHost().library.openFolder ? <FolderMenu /> : <LibraryMenu />}
        {!folderOnly && <FileMetaPill />}
      </div>
      {!folderOnly && <ZoomPill />}
    </div>
  );
}
