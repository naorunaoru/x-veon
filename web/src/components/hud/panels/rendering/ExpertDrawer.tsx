import { ChevronDown } from 'lucide-react';
import { useGrading } from '@/hooks/useGrading';
import type { OpenDrtConfig } from '@/gl/opendrt-params';
import { Slider } from '../../Slider';
import { Toggle } from '../../Toggle';
import { EXPERT_GROUPS } from './constants';

type Grading = ReturnType<typeof useGrading>;
type DrtKey = keyof OpenDrtConfig;

interface Props {
  g: Grading;
  open: boolean;
  setOpen: (open: boolean) => void;
}

/** The raw OpenDRT parameters ("stickshift") — every knob by its real name. */
export function ExpertDrawer({ g, open, setOpen }: Props) {
  return (
    <div className="xv-expert">
      <button type="button" className="xv-expert__head" onClick={() => setOpen(!open)}>
        <ChevronDown size={13} className={`xv-expert__chevron${open ? ' is-open' : ''}`} />
        <span className="xv-expert__title">Expert</span>
        <span className="xv-expert__tag">stickshift</span>
        <span className="xv-expert__aside">{open ? 'hide' : 'all params'}</span>
      </button>
      {open && (
        <div className="xv-expert__body">
          <p className="xv-expert__explainer">
            Every underlying OpenDRT knob, by its real name. Unwieldy by design — the
            controls above drive these for you. Reach in only when you need to.
          </p>
          {EXPERT_GROUPS.map((grp) => (
            <div key={grp.title} className="xv-expert__group">
              <div className="xv-expert__grouptitle">{grp.title}</div>
              {grp.rows.map((r) => (
                r.kind === 'toggle' ? (
                  <div key={r.key} className="xv-toggle-row xv-expert__row">
                    <div>
                      <span className="xv-toggle-row__label">{r.label}</span>
                      <div className="xv-expert__key">{r.key}</div>
                    </div>
                    <Toggle label={r.label}
                      checked={g.effective(r.key) as boolean}
                      onChange={(v) => g.setDrt(r.key, v as OpenDrtConfig[DrtKey])} />
                  </div>
                ) : (
                  <div key={r.key} className="xv-expert__row">
                    <Slider label={r.label} min={r.min} max={r.max} step={r.step}
                      value={g.effective(r.key) as number} defaultValue={g.baseConfig[r.key] as number}
                      onChange={(v) => g.setDrt(r.key, v as OpenDrtConfig[DrtKey])} />
                    <div className="xv-expert__key">{r.key}</div>
                  </div>
                )
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
