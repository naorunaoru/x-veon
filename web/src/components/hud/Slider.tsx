import * as React from 'react';
import * as SliderPrimitive from '@radix-ui/react-slider';
import { isModified, formatDelta } from '@/renderer/grading/param-model';
import './Slider.css';

export interface SliderProps {
  label: string;
  value: number;
  defaultValue: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  /** Formats the numeric readout + delta. Defaults to the step's decimal precision. */
  format?: (v: number) => string;
  /** Optional CSS gradient for the track (e.g. temperature). Suppresses the delta fill. */
  gradientTrack?: string;
  /** Accent color when modified (e.g. a channel color). Defaults to primary. */
  accentColor?: string;
  /** Optional unit suffix appended to the absolute readout. */
  unit?: string;
  /** Optional override for the displayed readout (e.g. "5500K"). */
  infoLabel?: string;
  /** Greys the control and ignores input. */
  disabled?: boolean;
}

/** Decimal places implied by a slider step (e.g. 0.001 → 3, 1 → 0). */
function stepDecimals(step: number): number {
  if (!Number.isFinite(step) || step <= 0) return 2;
  const s = String(step);
  const dot = s.indexOf('.');
  return dot === -1 ? 0 : s.length - dot - 1;
}

const clamp01 = (n: number) => Math.min(1, Math.max(0, n));

export function Slider({
  label, value, defaultValue, min, max, step, onChange,
  format, gradientTrack, accentColor, unit = '', infoLabel, disabled = false,
}: SliderProps) {
  // Default the readout precision to the step so a sub-0.01 step never shows a
  // "modified" delta of +0.00 (isModified uses a 1e-6 epsilon).
  const fmt = format ?? ((v: number) => v.toFixed(stepDecimals(step)));
  const modified = isModified(value, defaultValue);
  const t = clamp01((value - min) / (max - min));
  const td = clamp01((defaultValue - min) / (max - min));
  const fillFrom = Math.min(t, td);
  const fillTo = Math.max(t, td);

  const rootStyle = {
    '--xv-slider-accent': accentColor,
    '--xv-track-bg': gradientTrack,
  } as React.CSSProperties;

  return (
    <div className={`xv-slider${modified ? ' is-modified' : ''}${disabled ? ' is-disabled' : ''}`} style={rootStyle}>
      <div className="xv-slider__head">
        <span className="xv-slider__label">{label}</span>
        <span className="xv-slider__readout">
          {modified && <span className="xv-slider__delta">{formatDelta(value, defaultValue, fmt)}</span>}
          <span>{infoLabel ?? `${fmt(value)}${unit}`}</span>
        </span>
      </div>
      <SliderPrimitive.Root
        className="xv-slider__root"
        min={min} max={max} step={step}
        value={[value]}
        disabled={disabled}
        onValueChange={([v]) => { if (!disabled) onChange(v); }}
      >
        <SliderPrimitive.Track className="xv-slider__track">
          <SliderPrimitive.Range className="xv-slider__range" />
        </SliderPrimitive.Track>
        {/* Tick + delta-fill overlay the track but live on the Root, not inside
            Radix's Track (whose only documented child is Range). Root is
            position:relative and the Track spans its full width, so the
            percentage offsets align. */}
        {!gradientTrack && modified && (
          <span
            className="xv-slider__fill"
            style={{ left: `${fillFrom * 100}%`, width: `${(fillTo - fillFrom) * 100}%` }}
          />
        )}
        <span className="xv-slider__tick" style={{ left: `${td * 100}%` }} />
        <SliderPrimitive.Thumb className="xv-slider__thumb" aria-label={label} />
      </SliderPrimitive.Root>
    </div>
  );
}
