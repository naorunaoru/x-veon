import { FloatingPanel } from '../FloatingPanel';
import { ScopeHistogram } from './ScopeHistogram';
import { useAppStore } from '@/store';
import './Scopes.css';
import './Panels.css';

export function ScopesPanel() {
  const setOpenPanel = useAppStore((s) => s.setOpenPanel);
  const showClipMask = useAppStore((s) => s.showClipMask);
  const setShowClipMask = useAppStore((s) => s.setShowClipMask);

  return (
    <FloatingPanel title="Scopes" onClose={() => setOpenPanel(null)}>
      <ScopeHistogram />
      <div className="xv-toggle-row">
        <span className="xv-toggle-row__label">Highlight clipping</span>
        <button
          className={`xv-toggle${showClipMask ? ' is-on' : ''}`}
          aria-pressed={showClipMask}
          aria-label="Toggle highlight clipping overlay"
          onClick={() => setShowClipMask(!showClipMask)}
        >
          {showClipMask ? 'On' : 'Off'}
        </button>
      </div>
    </FloatingPanel>
  );
}
