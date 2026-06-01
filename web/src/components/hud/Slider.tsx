import * as React from 'react';
import * as SliderPrimitive from '@radix-ui/react-slider';
import { isModified, formatDelta } from '@/lib/grading/param-model';
import './Slider.css';

export interface SliderProps {
  label: string;
  value: number;
  defaultValue: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  /** Formats the numeric readout + delta. Defaults to 2-dp fixed. */
  format?: (v: number) => string;
  /** Optional CSS gradient for the track (e.g. temperature). Suppresses the delta fill. */
  gradientTrack?: string;
  /** Accent color when modified (e.g. a channel color). Defaults to primary. */
  accentColor?: string;
  /** Optional unit suffix appended to the absolute readout. */
  unit?: string;
  /** Optional override for the displayed readout (e.g. "5500K"). */
  infoLabel?: string;
}

export function Slider({
  label, value, defaultValue, min, max, step, onChange,
  format = (v) => v.toFixed(2),
  gradientTrack, accentColor, unit = '', infoLabel,
}: SliderProps) {
  const modified = isModified(value, defaultValue);
  const t = (value - min) / (max - min);
  const td = (defaultValue - min) / (max - min);
  const fillFrom = Math.min(t, td);
  const fillTo = Math.max(t, td);

  const rootStyle = {
    '--xv-slider-accent': accentColor,
    '--xv-track-bg': gradientTrack,
  } as React.CSSProperties;

  return (
    <div className={`xv-slider${modified ? ' is-modified' : ''}`} style={rootStyle}>
      <div className="xv-slider__head">
        <span className="xv-slider__label">{label}</span>
        <span className="xv-slider__readout">
          {modified && <span className="xv-slider__delta">{formatDelta(value, defaultValue, format)}</span>}
          <span>{infoLabel ?? `${format(value)}${unit}`}</span>
        </span>
      </div>
      <SliderPrimitive.Root
        className="xv-slider__root"
        min={min} max={max} step={step}
        value={[value]}
        onValueChange={([v]) => onChange(v)}
      >
        <SliderPrimitive.Track className="xv-slider__track">
          {!gradientTrack && modified && (
            <span
              className="xv-slider__fill"
              style={{ left: `${fillFrom * 100}%`, width: `${(fillTo - fillFrom) * 100}%` }}
            />
          )}
          <span className="xv-slider__tick" style={{ left: `${td * 100}%` }} />
          <SliderPrimitive.Range className="xv-slider__range" />
        </SliderPrimitive.Track>
        <SliderPrimitive.Thumb className="xv-slider__thumb" aria-label={label} />
      </SliderPrimitive.Root>
    </div>
  );
}
