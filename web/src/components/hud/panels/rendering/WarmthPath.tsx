import * as React from 'react';
import { useGrading } from '@/app/hooks/useGrading';
import { useDrag, relPos, clamp } from './hooks';
import { Readout } from './Readout';

type Grading = ReturnType<typeof useGrading>;

const W = 348, H = 92, PAD = 4;
const GW = W - PAD * 2;
const AMBER = '#E8A438';

/** Creative-white-over-luminance: warm the highlights, keep shadows neutral. */
export function WarmthPath({ g }: { g: Grading }) {
  const svgRef = React.useRef<SVGSVGElement>(null);
  const cwp = g.effective('cwp');
  const cwp_rng = g.effective('cwp_rng');

  const onsetT = 1 - cwp_rng; // 0 = shadows .. 1 = highlights
  const onsetX = PAD + onsetT * GW;
  const amtH = cwp * (H - 20);

  const dragOnset = useDrag((e) => {
    const { x } = relPos(e, svgRef.current!);
    const t = clamp((x - PAD) / GW, 0, 1);
    g.setDrt('cwp_rng', +(1 - t).toFixed(2));
  });
  const dragAmt = useDrag((e) => {
    const { y } = relPos(e, svgRef.current!);
    const t = clamp(1 - (y - 6) / (H - 20), 0, 1);
    g.setDrt('cwp', +t.toFixed(2));
  });

  return (
    <div>
      <svg ref={svgRef} className="xv-rsvg" width={W} height={H}>
        <defs>
          <linearGradient id="xv-warmramp" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#0a0a0a" /><stop offset="100%" stopColor="#ededed" />
          </linearGradient>
          <linearGradient id="xv-warmtint" x1="0" y1="0" x2="1" y2="0">
            <stop offset={`${onsetT * 100}%`} stopColor="rgba(232,164,56,0)" />
            <stop offset="100%" stopColor={`rgba(232,164,56,${0.18 + cwp * 0.55})`} />
          </linearGradient>
        </defs>
        <rect x={PAD} y={6} width={GW} height={H - 26} rx={3} fill="url(#xv-warmramp)" />
        <rect x={PAD} y={6} width={GW} height={H - 26} rx={3} fill="url(#xv-warmtint)" />
        <line x1={onsetX} y1={2} x2={onsetX} y2={H - 18} stroke="var(--xv-foreground)" strokeWidth={1} opacity={0.7} strokeDasharray="2 2" />
        <g onPointerDown={dragOnset.start} style={{ cursor: 'ew-resize' }}>
          <circle cx={onsetX} cy={H - 18} r={11} fill="transparent" />
          <circle cx={onsetX} cy={H - 18} r={dragOnset.active ? 6 : 5} fill="var(--xv-foreground)" stroke="rgba(0,0,0,0.5)" />
        </g>
        <g onPointerDown={dragAmt.start} style={{ cursor: 'ns-resize' }}>
          <circle cx={W - PAD - 6} cy={(H - 20) - amtH + 6} r={11} fill="transparent" />
          <circle cx={W - PAD - 6} cy={(H - 20) - amtH + 6} r={dragAmt.active ? 6 : 5} fill={AMBER} stroke="rgba(0,0,0,0.5)" />
        </g>
        <text x={PAD} y={H - 4} className="xv-rsvg__axis">shadows</text>
        <text x={W - PAD} y={H - 4} textAnchor="end" className="xv-rsvg__axis">highlights</text>
      </svg>
      <div className="xv-readout-grid xv-readout-grid--2">
        <Readout label="Warmth" value={cwp === 0 ? 'D65' : cwp >= 0.99 ? 'D50' : cwp.toFixed(2)}
          accent={cwp > 0 ? AMBER : undefined}
          scrub={{ get: () => g.effective('cwp'), set: (v) => g.setDrt('cwp', +v.toFixed(2)), min: 0, max: 1 }} />
        <Readout label="Onset" value={cwp_rng === 0 ? 'off' : `${Math.round((1 - onsetT) * 100)}%`}
          accent={cwp_rng > 0 ? 'var(--xv-primary)' : undefined}
          scrub={{ get: () => g.effective('cwp_rng'), set: (v) => g.setDrt('cwp_rng', +v.toFixed(2)), min: 0, max: 1 }} />
      </div>
    </div>
  );
}
