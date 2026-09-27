import { useId, useMemo, useState } from 'react';
import { RotateCcw, Undo2, X } from 'lucide-react';
import { useGrading } from '@/app/hooks/useGrading';
import { useAppStore } from '@/app/store';
import { configWithOverrides, LOOK_PRESETS, type OpenDrtConfig } from '@/renderer/grading/opendrt-params';
import { RENDERING_KEYS, modifiedDrtKeys } from '@/renderer/grading/sections';
import { RSection } from './rendering/RSection';
import { TonescaleGraph } from './rendering/TonescaleGraph';
import { ColorWheelControl } from './rendering/ColorWheelControl';
import { WarmthPath } from './rendering/WarmthPath';
import { PurityCurve } from './rendering/PurityCurve';
import { ExpertDrawer } from './rendering/ExpertDrawer';
import type { LookPreset } from '@/lib/types';
import './Panels.css';
import './RenderingPanel.css';

type DrtKey = keyof OpenDrtConfig;

export function RenderingPanel({ embedded = false }: { embedded?: boolean }) {
  const g = useGrading();
  const setOpenPanel = useAppStore((s) => s.setOpenPanel);
  const [expert, setExpert] = useState(false);
  const lookId = useId();
  const changed = useMemo(() => modifiedDrtKeys(g.baseConfig, g.overrides), [g.baseConfig, g.overrides]);
  const modified = changed.length > 0;

  const cfg = useMemo(
    () => configWithOverrides(g.baseConfig, g.overrides, g.preOverrides),
    [g.baseConfig, g.overrides, g.preOverrides],
  );

  const isMod = (keys: DrtKey[]) => keys.some((k) => changed.includes(k));
  const resetKeys = (keys: DrtKey[]) => g.resetSection(keys, []);

  return (
    <div className={embedded ? "xv-adjustment-group" : "xv-rpanel xv-glass-heavy"}>
      <header className="xv-rpanel__header">
        <span className="xv-rpanel__title">Rendering</span>
        <span className="xv-rpanel__tag">OpenDRT</span>
        {modified && <span className="xv-rpanel__dot" />}
        <div className="xv-rpanel__actions">
          {!embedded && <button type="button" className="xv-rpanel__icon" aria-label="Close panel" onClick={() => setOpenPanel(null)}>
            <X size={12} />
          </button>}
        </div>
      </header>

      <div className="xv-rpanel__body">
        <div className="xv-look">
          <label className="xv-rsection__sublabel" htmlFor={lookId}>Look</label>
          <select id={lookId} className="xv-select" value={g.lookPreset} disabled={!g.fileId}
            onChange={(e) => g.setLook(e.target.value as LookPreset)}>
            {(Object.entries(LOOK_PRESETS) as [LookPreset, typeof LOOK_PRESETS[LookPreset]][]).map(([id, look]) => (
              <option key={id} value={id}>
                {look.label}{id === g.lookPreset && modified ? ' · Modified' : ''}
              </option>
            ))}
          </select>
          {(modified || g.canUndoLook) && (
            <div className="xv-look__actions">
              {g.canUndoLook && (
                <button type="button" className="xv-look__action" onClick={g.undoLook}>
                  <Undo2 size={12} /> Undo look change
                </button>
              )}
              {modified && (
                <button type="button" className="xv-look__action xv-look__reset" onClick={() => g.setLook(g.lookPreset)}>
                  <RotateCcw size={12} /> Reset look
                </button>
              )}
            </div>
          )}
        </div>

        <RSection title="Tone" modified={isMod(RENDERING_KEYS.tone)} onReset={() => resetKeys(RENDERING_KEYS.tone)}>
          <TonescaleGraph g={g} cfg={cfg} />
        </RSection>

        <RSection title="Colour" modified={isMod(RENDERING_KEYS.colour)} onReset={() => resetKeys(RENDERING_KEYS.colour)}>
          <ColorWheelControl g={g} />
          <div className="xv-colour-group">
            <div className="xv-rsection__sublabel">Purity</div>
            <PurityCurve g={g} />
          </div>
          <div className="xv-colour-group">
            <div className="xv-rsection__sublabel">Highlight warmth</div>
            <WarmthPath g={g} />
          </div>
        </RSection>

        <ExpertDrawer g={g} open={expert} setOpen={setExpert} />
      </div>
    </div>
  );
}
