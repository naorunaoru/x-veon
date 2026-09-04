import * as React from 'react';
import { useGrading } from '@/hooks/useGrading';
import type { OpenDrtConfig } from '@/renderer/grading/opendrt-params';
import { Slider } from '../../Slider';
import { useDrag, relPos, clamp } from './hooks';
import { HueChip } from './Readout';
import {
  HUES, WHEEL_MODES, RANGE_KEY, ptToWheel, wheelToPt,
  type Hue, type WheelMode,
} from './constants';

type Grading = ReturnType<typeof useGrading>;
type DrtKey = keyof OpenDrtConfig;

const SZ = 230, CX = SZ / 2, CY = SZ / 2;
const R_BASE = 64, R_MIN = 34, R_MAX = 96;
const fmt = (v: number) => (v > 0 ? '+' : '') + v.toFixed(2);

export function ColorWheelControl({ g }: { g: Grading }) {
  const svgRef = React.useRef<SVGSVGElement>(null);
  const [mode, setMode] = React.useState<WheelMode>('brl');
  const [hover, setHover] = React.useState<string | null>(null);
  const [dragKey, setDragKey] = React.useState<string | null>(null);
  const dragHue = React.useRef<Hue | null>(null);

  // ── per-mode read/write through the real OpenDRT keys ────────────────────
  const readNode = React.useCallback((h: Hue): number => {
    const key = h.keys[mode];
    if (!key) return 0;
    const v = g.effective(key) as number;
    if (mode === 'pur') return clamp(ptToWheel(v, g.baseConfig[key] as number), -1, 1);
    return clamp(v, -1, 1);
  }, [g, mode]);

  const writeNode = React.useCallback((h: Hue, v: number) => {
    const key = h.keys[mode];
    if (!key) return;
    if (mode === 'pur') g.setDrt(key, wheelToPt(v, g.baseConfig[key] as number) as OpenDrtConfig[DrtKey]);
    else g.setDrt(key, +v.toFixed(2) as OpenDrtConfig[DrtKey]);
  }, [g, mode]);

  // node screen position for the current value
  const nodePos = (h: Hue) => {
    const v = readNode(h);
    if (mode === 'hue') {
      const a = (h.deg + v * 22 - 90) * Math.PI / 180;
      return { x: CX + Math.cos(a) * R_BASE, y: CY + Math.sin(a) * R_BASE, a };
    }
    const r = R_BASE + v * 30; // outward = +brilliance / +purity
    const a = (h.deg - 90) * Math.PI / 180;
    return { x: CX + Math.cos(a) * r, y: CY + Math.sin(a) * r, a };
  };

  const drag = useDrag((e) => {
    const h = dragHue.current;
    if (!h || !svgRef.current) return;
    const { x, y } = relPos(e, svgRef.current);
    const dx = x - CX, dy = y - CY;
    if (mode === 'hue') {
      const ang = Math.atan2(dy, dx) * 180 / Math.PI + 90;
      let delta = ang - h.deg;
      while (delta > 180) delta -= 360;
      while (delta < -180) delta += 360;
      writeNode(h, clamp(+(delta / 22).toFixed(2), -1, 1));
    } else {
      const r = Math.sqrt(dx * dx + dy * dy);
      writeNode(h, clamp(+((r - R_BASE) / 30).toFixed(2), -1, 1));
    }
  });
  React.useEffect(() => {
    if (!drag.active) { dragHue.current = null; setDragKey(null); }
  }, [drag.active]);

  const startNode = (h: Hue) => (e: React.PointerEvent) => {
    if (mode === 'pur' && !h.keys.pur) return; // dimmed secondaries are inert
    dragHue.current = h;
    setDragKey(h.k);
    drag.start(e);
  };

  const poly = mode === 'hue' ? null
    : HUES.map((h) => { const p = nodePos(h); return `${p.x.toFixed(1)},${p.y.toFixed(1)}`; }).join(' ');

  const range = RANGE_KEY[mode];

  return (
    <div>
      <div className="xv-modeswitch">
        {WHEEL_MODES.map((m) => (
          <button key={m.k} type="button"
            className={`xv-modeswitch__btn${mode === m.k ? ' is-active' : ''}`}
            onClick={() => setMode(m.k)}>
            {m.label}
          </button>
        ))}
      </div>

      <svg ref={svgRef} className="xv-rsvg xv-wheel" width={SZ} height={SZ}>
        {/* hue wedge backdrops */}
        {HUES.map((h) => {
          const a0 = (h.deg - 30 - 90) * Math.PI / 180, a1 = (h.deg + 30 - 90) * Math.PI / 180;
          const x0 = CX + Math.cos(a0) * R_MAX, y0 = CY + Math.sin(a0) * R_MAX;
          const x1 = CX + Math.cos(a1) * R_MAX, y1 = CY + Math.sin(a1) * R_MAX;
          const xi0 = CX + Math.cos(a0) * R_MIN, yi0 = CY + Math.sin(a0) * R_MIN;
          const xi1 = CX + Math.cos(a1) * R_MIN, yi1 = CY + Math.sin(a1) * R_MIN;
          return (
            <path key={h.k}
              d={`M ${x0} ${y0} A ${R_MAX} ${R_MAX} 0 0 1 ${x1} ${y1} L ${xi1} ${yi1} A ${R_MIN} ${R_MIN} 0 0 0 ${xi0} ${yi0} Z`}
              fill={h.col} opacity={hover === h.k ? 0.2 : 0.1} />
          );
        })}
        <circle cx={CX} cy={CY} r={R_BASE} fill="none" stroke="var(--xv-border)" strokeWidth={1} strokeDasharray="2 3" />
        {HUES.map((h) => {
          const a = (h.deg - 90) * Math.PI / 180;
          return (
            <line key={h.k} x1={CX + Math.cos(a) * R_MIN} y1={CY + Math.sin(a) * R_MIN}
              x2={CX + Math.cos(a) * R_MAX} y2={CY + Math.sin(a) * R_MAX}
              stroke="var(--xv-border)" strokeWidth={1} opacity={0.4} />
          );
        })}
        {poly && <polygon points={poly} fill="oklch(0.72 0.12 220 / 0.08)" stroke="var(--xv-primary)" strokeWidth={1.5} strokeLinejoin="round" />}
        {/* hue-twist tangent arrows */}
        {mode === 'hue' && HUES.map((h) => {
          const v = readNode(h);
          if (Math.abs(v) < 0.02) return null;
          const base = (h.deg - 90) * Math.PI / 180;
          const x0 = CX + Math.cos(base) * (R_BASE + 16), y0 = CY + Math.sin(base) * (R_BASE + 16);
          const p = nodePos(h);
          return <line key={h.k} x1={x0} y1={y0} x2={p.x + Math.cos(p.a) * 14} y2={p.y + Math.sin(p.a) * 14}
            stroke={h.col} strokeWidth={1.5} opacity={0.8} />;
        })}
        {/* nodes */}
        {HUES.map((h) => {
          const p = nodePos(h);
          const v = readNode(h);
          const active = dragKey === h.k;
          const dim = mode === 'pur' && !h.keys.pur;
          const set = Math.abs(v) > 0.02;
          return (
            <g key={h.k} onPointerDown={startNode(h)}
              onPointerEnter={() => setHover(h.k)} onPointerLeave={() => setHover(null)}
              style={{ cursor: dim ? 'default' : 'grab', opacity: dim ? 0.4 : 1 }}>
              <circle cx={p.x} cy={p.y} r={12} fill="transparent" />
              <circle cx={p.x} cy={p.y} r={active ? 8 : 6.5} fill={h.col}
                stroke={set ? 'var(--xv-foreground)' : 'rgba(0,0,0,0.5)'} strokeWidth={set ? 1.5 : 1} />
              {(active || hover === h.k) && !dim && (
                <text x={p.x} y={p.y - 13} textAnchor="middle" className="xv-rsvg__handle-label">{fmt(v)}</text>
              )}
            </g>
          );
        })}
        <text x={CX} y={CY - 4} textAnchor="middle" className="xv-wheel__center">{mode.toUpperCase()}</text>
        <text x={CX} y={CY + 8} textAnchor="middle" className="xv-wheel__center is-dim">{HUES.map((h) => h.k).join(' ')}</text>
      </svg>

      <div className="xv-huerow">
        {HUES.map((h) => {
          const inert = mode === 'pur' && !h.keys.pur;
          return (
            <HueChip key={h.k} hue={h} value={readNode(h)}
              onScrub={(nv) => { if (!inert) writeNode(h, clamp(+nv.toFixed(2), -1, 1)); }} />
          );
        })}
      </div>

      <div className="xv-wheel__range">
        <Slider label="Intensity range" min={range.min} max={range.max} step={0.01}
          value={g.effective(range.key) as number} defaultValue={g.baseConfig[range.key] as number}
          onChange={(v) => g.setDrt(range.key, v as OpenDrtConfig[DrtKey])} />
      </div>
    </div>
  );
}
