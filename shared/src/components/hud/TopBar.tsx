import { StatusPill } from './StatusPill';
import { FileMetaPill } from './FileMetaPill';
import { ZoomPill } from './ZoomPill';
import './TopBar.css';

export function TopBar() {
  return (
    <div className="xv-topbar">
      <div className="xv-topbar__left">
        <StatusPill />
        <FileMetaPill />
      </div>
      <ZoomPill />
    </div>
  );
}
