import { StatusPill } from './StatusPill';
import './TopBar.css';

/**
 * Phase 1 top bar: status pill only. File-meta pill, processing pill, and zoom
 * pill are added in later phases.
 */
export function TopBar() {
  return (
    <div className="xv-topbar">
      <StatusPill />
    </div>
  );
}
