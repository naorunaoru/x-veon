import { useAppStore } from '@/app/store';
import type { HistogramChannel } from '@/renderer';
import './panels/Scopes.css';
const SOURCES = [{ id: 'display', label: 'Display' }, { id: 'scene', label: 'Scene' }] as const;
const CHANNELS: { id: HistogramChannel; label: string }[] = [{ id: 'luma', label: 'L' }, { id: 'rgb', label: 'RGB' }, { id: 'ev', label: 'EV' }];
export function ScopeControls() {
  const source = useAppStore((s) => s.histogramSource);
  const channel = useAppStore((s) => s.histogramChannel);
  const setSource = useAppStore((s) => s.setHistogramSource);
  const setChannel = useAppStore((s) => s.setHistogramChannel);
  return (
        <div className="xv-modes">
          <div className="xv-modes__grp">
            {SOURCES.map((s) => (
              <button key={s.id} className={`xv-modes__btn${source === s.id ? ' is-active' : ''}`}
                aria-pressed={source === s.id} onClick={() => setSource(s.id)}>{s.label}</button>
            ))}
          </div>
          <div className="xv-modes__grp">
            {CHANNELS.map((c) => (
              <button key={c.id} className={`xv-modes__btn${channel === c.id ? ' is-active' : ''}`}
                aria-label={c.id === 'luma' ? 'Luminance' : c.label} aria-pressed={channel === c.id} onClick={() => setChannel(c.id)}>{c.label}</button>
            ))}
          </div>
        </div>);
}
