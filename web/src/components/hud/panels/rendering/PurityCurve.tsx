import * as React from 'react';
import { useGrading } from '@/app/hooks/useGrading';
import { OPENDRT_LIMITS } from '@/renderer/grading/opendrt-params';
import { Slider } from '../../Slider';
import { useDrag, relPos, clamp } from './hooks';

type Grading = ReturnType<typeof useGrading>;

const W = 348, H = 108, PAD = 4, PADB = 16;
const GW = W - PAD * 2, GH = H - PAD - PADB;
const AMBER = '#E8A438';

const xOf = (t: number) => PAD + t * GW;
const yOf = (v: number) => PAD + (1 - v) * GH;

/** Saturation across the tonal range. Envelope is a visual approximation;
 *  handles drive the real rs_sa / ptm_low / pt_rng_high keys. */
export function PurityCurve({ g }: { g: Grading }) {
  const svgRef = React.useRef<SVGSVGElement>(null);
  const sat = g.effective('rs_sa');
  const mid = g.effective('ptm_low');
  const hi = g.effective('pt_rng_high');

  const yAt = React.useCallback((t: number) => {
    const base = 0.45 + sat * 0.5;
    const midBump = mid * 0.35 * Math.exp(-Math.pow((t - 0.45) * 3, 2));
    const hiRoll = (1 - hi) * 0.6 * Math.pow(clamp((t - 0.4) / 0.6, 0, 1), 1.6);
    return clamp(base + midBump - hiRoll, 0.05, 1);
  }, [sat, mid, hi]);

  const path = React.useMemo(() => {
    const N = 60;
    let d = '';
    for (let i = 0; i <= N; i++) { const t = i / N; d += `${i === 0 ? 'M' : 'L'} ${xOf(t).toFixed(1)} ${yOf(yAt(t)).toFixed(1)} `; }
    return d;
  }, [yAt]);
  const area = `${path} L ${xOf(1)} ${yOf(0)} L ${xOf(0)} ${yOf(0)} Z`;

  const dragMid = useDrag((e) => {
    const { y } = relPos(e, svgRef.current!);
    const v = clamp((yOf(yAt(0.45)) - y) / 30 + mid, -1, 1);
    g.setDrt('ptm_low', +v.toFixed(2));
  });
  const dragHi = useDrag((e) => {
    const { x } = relPos(e, svgRef.current!);
    g.setDrt('pt_rng_high', +(0.25 + clamp((x - PAD) / GW, 0, 1) * 1.75).toFixed(2));
  });

  const midPt = { x: xOf(0.45), y: yOf(yAt(0.45)) };
  const hiX = clamp((hi - 0.25) / 1.75, 0, 1);
  const hiPt = { x: xOf(hiX), y: yOf(yAt(hiX)) };

  return (
    <div>
      <Slider label="Overall purity" min={OPENDRT_LIMITS.rs_sa[0]} max={OPENDRT_LIMITS.rs_sa[1]} step={0.01}
        value={sat} defaultValue={g.baseConfig.rs_sa} onChange={(v) => g.setDrt('rs_sa', v)} />
      <svg ref={svgRef} className="xv-rsvg xv-purity-svg" width={W} height={H}>
        <g stroke="var(--xv-border)" strokeWidth={1} opacity={0.45}>
          <line x1={PAD} y1={PAD + 0.5 * GH} x2={W - PAD} y2={PAD + 0.5 * GH} />
        </g>
        <path d={area} fill="oklch(0.72 0.12 220 / 0.10)" />
        <path d={path} fill="none" stroke="var(--xv-primary)" strokeWidth={2} />
        <g onPointerDown={dragMid.start} style={{ cursor: 'ns-resize' }}>
          <circle cx={midPt.x} cy={midPt.y} r={11} fill="transparent" />
          <circle cx={midPt.x} cy={midPt.y} r={dragMid.active ? 6 : 4.5} fill="var(--xv-primary)" stroke="rgba(0,0,0,0.5)" />
        </g>
        <g onPointerDown={dragHi.start} style={{ cursor: 'ew-resize' }}>
          <circle cx={hiPt.x} cy={hiPt.y} r={11} fill="transparent" />
          <circle cx={hiPt.x} cy={hiPt.y} r={dragHi.active ? 6 : 4.5} fill={AMBER} stroke="rgba(0,0,0,0.5)" />
        </g>
        <text x={PAD} y={H - 3} className="xv-rsvg__axis">shadows</text>
        <text x={W - PAD} y={H - 3} textAnchor="end" className="xv-rsvg__axis">highlights</text>
      </svg>
    </div>
  );
}
