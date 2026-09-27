import { FloatingPanel } from '../FloatingPanel';
import { Slider } from '../Slider';
import { useGrading } from '@/app/hooks/useGrading';
import { useAppStore } from '@/app/store';
import { isSectionModified, SECTION_KEYS } from '@/renderer/grading/sections';
import './Panels.css';

const TEMP_GRADIENT = 'linear-gradient(to right, #5B8FC9, #E8A438)';
const TINT_GRADIENT = 'linear-gradient(to right, #C850C0, #4BA84D)';
const PRESETS: { label: string; cct: number }[] = [
  { label: 'Daylight', cct: 5500 },
  { label: 'Shade', cct: 7500 },
  { label: 'Tungsten', cct: 3200 },
];

export function WhiteBalancePanel({ embedded = false }: { embedded?: boolean }) {
  const g = useGrading();
  const setOpenPanel = useAppStore((s) => s.setOpenPanel);
  const keys = SECTION_KEYS.whiteBalance!;
  const modified = isSectionModified('whiteBalance', g.overrides, g.preOverrides);
  const ready = !!g.shootingInfo;

  const tempK = g.shootingInfo?.tempK ?? 5500;
  const baseTempK = g.shootingInfo?.baseTempK ?? 5500;
  const tintValue = g.shootingInfo?.tintValue ?? 0;
  const baseTint = g.shootingInfo?.baseTint ?? 0;

  return (
    <FloatingPanel
      embedded={embedded}
      title="White balance"
      modified={modified}
      onReset={() => g.resetSection(keys.drt, keys.pre)}
      onClose={() => setOpenPanel(null)}
    >
      <Slider
        label="Temperature" min={2000} max={12000} step={50}
        value={tempK} defaultValue={baseTempK}
        onChange={g.handleTempChange}
        gradientTrack={TEMP_GRADIENT}
        infoLabel={`${tempK}K`}
      />
      <Slider
        label="Tint" min={-150} max={150} step={1}
        value={tintValue} defaultValue={baseTint}
        onChange={g.handleTintChange}
        gradientTrack={TINT_GRADIENT}
        infoLabel={`${tintValue > 0 ? '+' : ''}${tintValue}`}
      />
      <div className="xv-pgroup">
        <span className="xv-pgroup__title">Presets</span>
        <div className="xv-chiprow">
          <button className="xv-chip" disabled={!ready} onClick={() => g.resetSection([], ['wb_temp', 'wb_tint'])}>As shot</button>
          {PRESETS.map((p) => (
            <button key={p.label} className="xv-chip" disabled={!ready} onClick={() => g.handleTempChange(p.cct)}>
              {p.label}
            </button>
          ))}
        </div>
      </div>
    </FloatingPanel>
  );
}
