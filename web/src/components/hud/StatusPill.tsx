import { useAppStore } from '@/store';
import './StatusPill.css';

export function StatusPill() {
  const initialized = useAppStore((s) => s.initialized);
  const initError = useAppStore((s) => s.initError);
  const backend = useAppStore((s) => s.backend);
  const displayHdr = useAppStore((s) => s.displayHdr);

  if (initError) {
    return <div className="xv-status-pill xv-glass is-error xv-over-image">Init failed: {initError}</div>;
  }
  if (!initialized) {
    return (
      <div className="xv-status-pill xv-glass xv-over-image">
        <span className="xv-status-pill__spinner" />
        Loading models and WASM…
      </div>
    );
  }
  return (
    <div className="xv-status-pill xv-glass xv-over-image">
      {backend}{displayHdr ? ' · HDR' : ''}
    </div>
  );
}
