import { useMemo } from 'react';
import { FloatingPanel } from '../FloatingPanel';
import { Slider } from '../Slider';
import { ToneCurveViz } from './ToneCurveViz';
import { useGrading } from '@/hooks/useGrading';
import { useAppStore } from '@/store';
import { isSectionModified, SECTION_KEYS } from '@/lib/grading/sections';
import {
  configWithOverrides, computeTonescaleParams, TONESCALE_PRESETS,
  type OpenDrtConfig, type TonescalePreset,
} from '@/gl/opendrt-params';
import { sampleToneCurve } from '@/lib/grading/tonescale-curve';
import './Panels.css';

export function ToneCurvePanel() {
  const g = useGrading();
  const setOpenPanel = useAppStore((s) => s.setOpenPanel);
  const setFileOpenDrtOverride = useAppStore((s) => s.setFileOpenDrtOverride);
  const keys = SECTION_KEYS.toneCurve!;
  const modified = isSectionModified('toneCurve', g.overrides, g.preOverrides);

  const cfg = useMemo(
    () => configWithOverrides(g.baseConfig, g.overrides, g.preOverrides),
    [g.baseConfig, g.overrides, g.preOverrides],
  );
  const points = useMemo(() => sampleToneCurve(cfg, computeTonescaleParams(cfg), 64), [cfg]);

  const applyPreset = (key: string) => {
    if (!g.fileId || !key) return;
    const preset = TONESCALE_PRESETS[key as TonescalePreset];
    if (!preset) return;
    for (const [k, v] of Object.entries(preset.overrides) as [keyof OpenDrtConfig, number | boolean][]) {
      setFileOpenDrtOverride(g.fileId, k, v as OpenDrtConfig[typeof k]);
    }
  };

  return (
    <FloatingPanel
      title="Tone curve"
      modified={modified}
      onReset={() => g.resetSection(keys.drt, keys.pre)}
      onClose={() => setOpenPanel(null)}
    >
      <ToneCurveViz points={points} />
      <Slider
        label="Toe" min={0} max={0.02} step={0.001}
        value={g.effective('tn_toe')} defaultValue={g.baseConfig.tn_toe}
        onChange={(v) => g.setDrt('tn_toe', v)}
      />
      <Slider
        label="Shoulder" min={0} max={1} step={0.01}
        value={g.effective('tn_sh')} defaultValue={g.baseConfig.tn_sh}
        onChange={(v) => g.setDrt('tn_sh', v)}
      />
      <div className="xv-field">
        <label className="xv-field__label" htmlFor="xv-tonescale">Tonescale preset</label>
        <select
          id="xv-tonescale" className="xv-select" aria-label="Tonescale preset"
          value="" onChange={(e) => applyPreset(e.target.value)}
        >
          <option value="" disabled>Presets…</option>
          {Object.entries(TONESCALE_PRESETS).map(([key, { label }]) => (
            <option key={key} value={key}>{label}</option>
          ))}
        </select>
      </div>
    </FloatingPanel>
  );
}
