import { useScrub } from './hooks';
import type { Hue } from './constants';

interface ScrubSpec {
  get: () => number;
  set: (v: number) => void;
  min: number;
  max: number;
  pxFull?: number;
}

interface ReadoutProps {
  label: string;
  value: string;
  /** Accent applied to the value text + the active scrub ring. */
  accent?: string;
  /** Makes the chip DAW-scrubbable (press + drag ↕). */
  scrub?: ScrubSpec;
}

/** A compact label/value chip. With `scrub`, press and drag ↕ to nudge the value. */
export function Readout({ label, value, accent, scrub }: ReadoutProps) {
  const s = useScrub({
    getValue: scrub?.get ?? (() => 0),
    apply: scrub?.set ?? (() => {}),
    min: scrub?.min ?? 0,
    max: scrub?.max ?? 1,
    pxFull: scrub?.pxFull ?? 200,
  });
  return (
    <div
      className={`xv-readout-chip${scrub ? ' is-scrub' : ''}${s.active ? ' is-active' : ''}`}
      style={accent ? ({ '--xv-readout-accent': accent } as React.CSSProperties) : undefined}
      onPointerDown={scrub ? s.onPointerDown : undefined}
      title={scrub ? 'Drag ↕ to adjust · Shift = fine' : undefined}
    >
      <span className="xv-readout-chip__label">{label}</span>
      <span className="xv-readout-chip__value" style={accent ? { color: accent } : undefined}>{value}</span>
    </div>
  );
}

interface HueChipProps {
  hue: Hue;
  value: number;
  onScrub: (v: number) => void;
}

/** Per-hue value chip under the colour wheel — scrubs the active layer's value. */
export function HueChip({ hue, value, onScrub }: HueChipProps) {
  const s = useScrub({ getValue: () => value, apply: onScrub, min: -1, max: 1, pxFull: 150 });
  const fmt = (x: number) => (x > 0 ? '+' : '') + x.toFixed(2);
  return (
    <div
      className={`xv-huechip${s.active ? ' is-active' : ''}`}
      style={{ '--xv-readout-accent': hue.col } as React.CSSProperties}
      onPointerDown={s.onPointerDown}
      title="Drag ↕ to adjust · Shift = fine"
    >
      <span className="xv-huechip__dot" style={{ background: hue.col }} />
      <span className={`xv-huechip__val${Math.abs(value) > 0.02 ? ' is-set' : ''}`}>
        {value === 0 ? '·' : fmt(value)}
      </span>
    </div>
  );
}
