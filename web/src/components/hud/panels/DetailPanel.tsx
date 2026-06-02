import { FloatingPanel } from '../FloatingPanel';
import { Slider } from '../Slider';
import { useGrading } from '@/hooks/useGrading';
import { useAppStore } from '@/store';
import { isSectionModified, SECTION_KEYS } from '@/lib/grading/sections';
import './Panels.css';

/** Detail / sharpening. Minimal for now — richer detail controls land later. */
export function DetailPanel() {
  const g = useGrading();
  const setOpenPanel = useAppStore((s) => s.setOpenPanel);
  const keys = SECTION_KEYS.detail!;
  const modified = isSectionModified('detail', g.overrides, g.preOverrides);

  return (
    <FloatingPanel
      title="Detail"
      modified={modified}
      onReset={() => g.resetSection(keys.drt, keys.pre)}
      onClose={() => setOpenPanel(null)}
    >
      <span className="xv-readout">Unsharp-mask sharpening, applied before the rendering transform.</span>
      <Slider
        label="Sharpening" min={0} max={2} step={0.01}
        value={g.effectivePre('sharpen_amount')} defaultValue={0}
        onChange={(v) => g.setPre('sharpen_amount', v)}
      />
    </FloatingPanel>
  );
}
