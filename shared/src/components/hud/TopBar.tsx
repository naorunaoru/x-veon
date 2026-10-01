import { StatusPill } from './StatusPill';
import { FileMetaPill } from './FileMetaPill';
import { ZoomPill } from './ZoomPill';
import { FolderMenu } from './FolderMenu';
import './TopBar.css';

export function TopBar({ folderOnly = false }: { folderOnly?: boolean }) {
  return (
    <div className="xv-topbar">
      <div className="xv-topbar__left">
        <FolderMenu />
        {!folderOnly && <StatusPill />}
        {!folderOnly && <FileMetaPill />}
      </div>
      {!folderOnly && <ZoomPill />}
    </div>
  );
}
