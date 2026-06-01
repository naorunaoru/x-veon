import { StatusPill } from './StatusPill';
import { FileMetaPill } from './FileMetaPill';
import './TopBar.css';

export function TopBar() {
  return (
    <div className="xv-topbar">
      <StatusPill />
      <FileMetaPill />
    </div>
  );
}
