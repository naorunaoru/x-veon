import { FloatingPanel } from '../FloatingPanel';
import { Slider } from '../Slider';
import { useGrading } from '@/hooks/useGrading';
import { useAppStore } from '@/store';
import { isSectionModified, SECTION_KEYS } from '@/lib/grading/sections';
import './Panels.css';

const CHANNELS = [
  { key: 'brl_r' as const, label: 'Red', color: 'var(--xv-chan-r)' },
  { key: 'brl_g' as const, label: 'Green', color: 'var(--xv-chan-g)' },
  { key: 'brl_b' as const, label: 'Blue', color: 'var(--xv-chan-b)' },
];

export function BrilliancePanel() {
  const g = useGrading();
  const setOpenPanel = useAppStore((s) => s.setOpenPanel);
  const keys = SECTION_KEYS.brilliance!;
  const modified = isSectionModified('brilliance', g.overrides, g.preOverrides);

  return (
    <FloatingPanel
      title="Brilliance"
      modified={modified}
      onReset={() => g.resetSection(keys.drt, keys.pre)}
      onClose={() => setOpenPanel(null)}
    >
      <span className="xv-readout">Three-channel gain map — adds highlight brilliance without clipping.</span>
      {CHANNELS.map(({ key, label, color }) => (
        <Slider
          key={key}
          label={label} min={-1} max={1} step={0.01}
          value={g.effective(key)} defaultValue={g.baseConfig[key]}
          onChange={(v) => g.setDrt(key, v)}
          accentColor={color}
        />
      ))}
    </FloatingPanel>
  );
}
