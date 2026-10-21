import type { CSSProperties } from 'react';
import { FloatingPanel } from '../FloatingPanel';
import { Slider } from '../Slider';
import { Toggle } from '../Toggle';
import { useLens } from '@/app/hooks/useLens';
import { useAppStore } from '@/app/store';
import type { LensChoice } from '@/app/lens/lensfun';
import { DEFAULT_LENS_EDIT, lensSourceKey, type LensManual } from '@/lib/lens/edit';
import './Panels.css';

/** Nothing is editable until the photo has decoded. */
const FIELDS: CSSProperties = { border: 0, padding: 0, margin: 0, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 16 };

const choiceKey = ({ file, model }: LensChoice) => lensSourceKey({ lensfun: { file, model } });
/** "Make Model", unless the database's model name already starts with the make. */
const choiceLabel = ({ make, model }: LensChoice) => (model.toLowerCase().startsWith(make.toLowerCase()) ? model : `${make} ${model}`);

/** Lens correction: where the profile comes from, which corrections are on, and the manual sliders. */
export function LensPanel({ embedded = false }: { embedded?: boolean }) {
  const lens = useLens();
  const setOpenPanel = useAppStore((s) => s.setOpenPanel);
  const { edit, readouts, manualLive } = lens;

  // A hand-picked lens stays selectable while the list loads, or when the list lacks it.
  const picked = typeof edit.source === 'object' ? edit.source.lensfun : null;
  const choices = picked && !lens.lenses.some((c) => c.file === picked.file && c.model === picked.model)
    ? [{ file: picked.file, make: '', model: picked.model }, ...lens.lenses]
    : lens.lenses;

  function chooseSource(value: string) {
    if (value === 'auto' || value === 'manual') return lens.setSource(value);
    const choice = choices.find((c) => choiceKey(c) === value);
    if (choice) lens.setSource({ lensfun: { file: choice.file, model: choice.model } });
  }

  // The sliders are the correction in force only in Manual; otherwise they show the profile's effect at the corner.
  const shown: LensManual = manualLive ? edit.manual : { ...DEFAULT_LENS_EDIT.manual, ...readouts };

  return (
    <FloatingPanel
      embedded={embedded}
      title="Lens"
      modified={lens.modified}
      onReset={lens.reset}
      onClose={() => setOpenPanel(null)}
    >
      <fieldset disabled={!lens.ready} style={FIELDS}>
        <div className="xv-pgroup">
          <select
            className="xv-select"
            aria-label="Lens source"
            value={lensSourceKey(edit.source)}
            onChange={(e) => chooseSource(e.target.value)}
          >
            <option value="auto">Auto</option>
            <option value="manual">Manual</option>
            {choices.length > 0 && (
              <optgroup label="Lensfun">
                {choices.map((c) => <option key={choiceKey(c)} value={choiceKey(c)}>{choiceLabel(c)}</option>)}
              </optgroup>
            )}
          </select>
          <div>
            <div className="xv-readout">{lens.status.line}</div>
            {lens.status.detail && <div className="xv-readout">{lens.status.detail}</div>}
          </div>
        </div>

        <div className="xv-pgroup" role="group" aria-label="Distortion">
          <div className="xv-toggle-row">
            <span className="xv-pgroup__title">Distortion</span>
            <Toggle label="Distortion" checked={edit.distortion} onChange={(on) => lens.setToggle('distortion', on)} />
          </div>
          <Slider
            label="Amount" unit="%" min={-20} max={20} step={0.1} defaultValue={0}
            value={shown.distortion} disabled={!manualLive}
            onChange={(v) => lens.setManual('distortion', v)}
          />
        </div>

        <div className="xv-pgroup" role="group" aria-label="Chromatic aberration">
          <div className="xv-toggle-row">
            <span className="xv-pgroup__title">Chromatic aberration</span>
            <Toggle label="Chromatic aberration" checked={edit.ca} onChange={(on) => lens.setToggle('ca', on)} />
          </div>
          <Slider
            label="Red/cyan" unit="px" min={-10} max={10} step={0.1} defaultValue={0}
            value={shown.caRed} disabled={!manualLive}
            onChange={(v) => lens.setManual('caRed', v)}
          />
          <Slider
            label="Blue/yellow" unit="px" min={-10} max={10} step={0.1} defaultValue={0}
            value={shown.caBlue} disabled={!manualLive}
            onChange={(v) => lens.setManual('caBlue', v)}
          />
        </div>

        <div className="xv-pgroup" role="group" aria-label="Vignetting">
          <div className="xv-toggle-row">
            <span className="xv-pgroup__title">Vignetting</span>
            <Toggle label="Vignetting" checked={edit.vignetting} onChange={(on) => lens.setToggle('vignetting', on)} />
          </div>
          <Slider
            label="Amount" unit="EV" min={-3} max={3} step={0.05} defaultValue={0}
            value={shown.vignetting} disabled={!manualLive}
            onChange={(v) => lens.setManual('vignetting', v)}
          />
          <Slider
            label="Midpoint" min={0.1} max={0.9} step={0.01} defaultValue={0.5}
            value={shown.midpoint} disabled={!manualLive}
            onChange={(v) => lens.setManual('midpoint', v)}
          />
        </div>
      </fieldset>
    </FloatingPanel>
  );
}
