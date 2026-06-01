import { FloatingPanel } from '../FloatingPanel';
import { Slider } from '../Slider';
import { useGrading } from '@/hooks/useGrading';
import { useAppStore } from '@/store';
import { isSectionModified, SECTION_KEYS } from '@/lib/grading/sections';
import './Panels.css';

function formatEv(ev: number): string {
  return `${ev >= 0 ? '+' : ''}${ev.toFixed(2)} EV`;
}

export function ExposurePanel() {
  const g = useGrading();
  const setOpenPanel = useAppStore((s) => s.setOpenPanel);
  const keys = SECTION_KEYS.exposure!;
  const modified = isSectionModified('exposure', g.overrides, g.preOverrides);

  const exposureEv = g.shootingInfo?.exposureEv ?? g.effectivePre('exposure');
  const baseBias = g.shootingInfo?.baseBias ?? 0;

  return (
    <FloatingPanel
      title="Exposure"
      modified={modified}
      onReset={() => g.resetSection(keys.drt, keys.pre)}
      onClose={() => setOpenPanel(null)}
    >
      <Slider
        label="Exposure" min={-5} max={5} step={0.01}
        value={exposureEv} defaultValue={baseBias}
        onChange={g.handleExposureChange}
        infoLabel={formatEv(exposureEv)}
      />
      <Slider
        label="Contrast" min={0.5} max={2.5} step={0.01}
        value={g.effective('tn_con')} defaultValue={g.baseConfig.tn_con}
        onChange={(v) => g.setDrt('tn_con', v)}
      />
      <Slider
        label="Local contrast" min={0} max={2} step={0.01}
        value={g.effective('tn_lcon')} defaultValue={g.baseConfig.tn_lcon}
        onChange={(v) => g.setDrt('tn_lcon', v)}
      />
    </FloatingPanel>
  );
}
