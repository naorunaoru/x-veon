import { FloatingPanel } from '../FloatingPanel';
import { useGrading } from '@/hooks/useGrading';
import { useAppStore } from '@/store';
import type { LookPreset } from '@/pipeline/types';
import './Panels.css';

const LOOKS: { id: LookPreset; label: string }[] = [
  { id: 'default', label: 'Default' },
  { id: 'colorful', label: 'Colorful' },
  { id: 'umbra', label: 'Umbra' },
  { id: 'base', label: 'Base' },
  { id: 'flat', label: 'Flat' },
];

export function LooksPanel() {
  const g = useGrading();
  const setOpenPanel = useAppStore((s) => s.setOpenPanel);

  return (
    <FloatingPanel title="Looks" onClose={() => setOpenPanel(null)}>
      <div className="xv-cardlist">
        {LOOKS.map(({ id, label }) => {
          const selected = g.lookPreset === id;
          return (
            <button
              key={id}
              className={`xv-card${selected ? ' is-selected' : ''}`}
              onClick={() => g.setLook(id)}
            >
              {label}
              {selected && <span className="xv-card__dot" />}
            </button>
          );
        })}
      </div>
    </FloatingPanel>
  );
}
