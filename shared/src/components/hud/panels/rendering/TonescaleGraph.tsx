import * as React from 'react';
import { useGrading } from '@/app/hooks/useGrading';
import { computeTonescaleParams, OPENDRT_LIMITS, type GradingConfig } from '@/renderer/grading/opendrt-params';
import { evalTonescale } from '@/renderer/grading/tonescale-curve';
import { isModified } from '@/renderer/grading/param-model';
import { useRelativeDrag, clamp, lerp } from './hooks';
import { Readout } from './Readout';
import { Slider } from '../../Slider';

type Grading = ReturnType<typeof useGrading>;

const W = 348, H = 184, PADL = 4, PADR = 4, PADT = 8, PADB = 18;
const GW = W - PADL - PADR, GH = H - PADT - PADB;
const ST_MIN = -7, ST_MAX = 6;
// Stops at which each handle rides the curve.
const STOP_GREY = 0, STOP_CON = 1.6, STOP_TOE = -3.4, STOP_SH = 4.2;

const xOfStop = (st: number) => PADL + ((st - ST_MIN) / (ST_MAX - ST_MIN)) * GW;
const yOfDisp = (d: number) => PADT + (1 - clamp(d, 0, 1)) * GH;

interface Props {
  g: Grading;
  cfg: GradingConfig;
}

export function TonescaleGraph({ g, cfg }: Props) {
  const svgRef = React.useRef<SVGSVGElement>(null);
  const ts = React.useMemo(() => computeTonescaleParams(cfg), [cfg]);

  // Display code value for a given scene stop, from the real tonescale.
  const dispAt = React.useCallback(
    (stop: number) => evalTonescale(0.18 * Math.pow(2, stop), cfg, ts),
    [cfg, ts],
  );

  const path = React.useMemo(() => {
    const N = 80;
    let d = '';
    for (let i = 0; i <= N; i++) {
      const st = lerp(ST_MIN, ST_MAX, i / N);
      d += `${i === 0 ? 'M' : 'L'} ${xOfStop(st).toFixed(1)} ${yOfDisp(dispAt(st)).toFixed(1)} `;
    }
    return d;
  }, [dispAt]);
  const area = `${path} L ${xOfStop(ST_MAX).toFixed(1)} ${yOfDisp(0)} L ${xOfStop(ST_MIN).toFixed(1)} ${yOfDisp(0)} Z`;

  const hp = (stop: number) => ({ x: xOfStop(stop), y: yOfDisp(dispAt(stop)) });
  const gpt = hp(STOP_GREY), cpt = hp(STOP_CON), tpt = hp(STOP_TOE), spt = hp(STOP_SH);

  // Ranges follow the render-time limits, so the handles never reach values the shader clamps.
  const [lgMin, lgMax] = OPENDRT_LIMITS.tn_lg;
  const [conMin, conMax] = OPENDRT_LIMITS.tn_con;
  const [toeMin, toeMax] = OPENDRT_LIMITS.tn_toe;
  const SH_MIN = 0.15, SH_MAX = 0.95;
  const linear = (lo: number, hi: number) => ({
    toT: (v: number) => (v - lo) / (hi - lo),
    fromT: (t: number) => lerp(lo, hi, t),
  });

  const dragGrey = useRelativeDrag({
    axis: 'y', pixelsPerRange: -GH, get: () => g.effective('tn_lg'), ...linear(lgMin, lgMax),
    apply: (v) => g.setDrt('tn_lg', +v.toFixed(1)),
  });
  const dragCon = useRelativeDrag({
    axis: 'y', pixelsPerRange: -GH, get: () => g.effective('tn_con'), ...linear(conMin, conMax),
    apply: (v) => g.setDrt('tn_con', +v.toFixed(2)),
  });
  const dragToe = useRelativeDrag({
    axis: 'y', pixelsPerRange: GH, get: () => g.effective('tn_toe'),
    toT: (v) => Math.pow((v - toeMin) / (toeMax - toeMin), 1 / 1.4),
    fromT: (t) => lerp(toeMin, toeMax, Math.pow(t, 1.4)),
    apply: (v) => g.setDrt('tn_toe', +v.toFixed(4)),
  });
  const dragSh = useRelativeDrag({
    axis: 'x', pixelsPerRange: GW, get: () => g.effective('tn_sh'), ...linear(SH_MIN, SH_MAX),
    apply: (v) => g.setDrt('tn_sh', +v.toFixed(2)),
  });

  const Handle = ({ pt, on, axis, label }: {
    pt: { x: number; y: number };
    on: ReturnType<typeof useRelativeDrag>;
    axis: 'x' | 'y';
    label: string;
  }) => (
    <g onPointerDown={on.start} style={{ cursor: axis === 'x' ? 'ew-resize' : 'ns-resize' }}>
      <circle cx={pt.x} cy={pt.y} r={11} fill="transparent" />
      <circle cx={pt.x} cy={pt.y} r={on.active ? 6 : 4.5} fill="var(--xv-primary)" stroke="rgba(0,0,0,0.5)" strokeWidth={1} />
      {on.active && (
        <text x={pt.x} y={pt.y - 12} textAnchor="middle" className="xv-rsvg__handle-label">{label}</text>
      )}
    </g>
  );

  const tn_lg = g.effective('tn_lg'), tn_con = g.effective('tn_con');
  const tn_toe = g.effective('tn_toe'), tn_sh = g.effective('tn_sh');

  return (
    <div>
      <svg ref={svgRef} className="xv-rsvg" width={W} height={H}>
        <g stroke="var(--xv-border)" strokeWidth={1} opacity={0.5}>
          {[0.25, 0.5, 0.75].map((f) => (
            <line key={f} x1={PADL} y1={PADT + f * GH} x2={W - PADR} y2={PADT + f * GH} />
          ))}
          {[-5, -2.47, 0, 2, 4].map((st) => (
            <line key={st} x1={xOfStop(st)} y1={PADT} x2={xOfStop(st)} y2={PADT + GH} />
          ))}
        </g>
        {/* middle-grey guide (0 EV) */}
        <line x1={xOfStop(0)} y1={PADT} x2={xOfStop(0)} y2={PADT + GH}
          stroke="var(--xv-muted-foreground)" strokeWidth={1} strokeDasharray="2 3" opacity={0.6} />
        {/* identity reference */}
        <line x1={xOfStop(ST_MIN)} y1={yOfDisp(0)} x2={xOfStop(ST_MAX)} y2={yOfDisp(1)}
          stroke="var(--xv-fg-dim)" strokeWidth={1} strokeDasharray="2 4" opacity={0.5} />
        <path d={area} fill="oklch(0.72 0.12 220 / 0.10)" />
        <path d={path} fill="none" stroke="var(--xv-primary)" strokeWidth={2} />
        <text x={xOfStop(0)} y={H - 5} textAnchor="middle" className="xv-rsvg__axis is-grey">18%</text>
        <text x={xOfStop(-5)} y={H - 5} textAnchor="middle" className="xv-rsvg__axis">−5EV</text>
        <text x={xOfStop(4)} y={H - 5} textAnchor="middle" className="xv-rsvg__axis">+4EV</text>
        <Handle pt={tpt} on={dragToe} axis="y" label={`toe ${tn_toe.toFixed(3)}`} />
        <Handle pt={gpt} on={dragGrey} axis="y" label={`grey ${tn_lg.toFixed(1)}`} />
        <Handle pt={cpt} on={dragCon} axis="y" label={`con ${tn_con.toFixed(2)}`} />
        <Handle pt={spt} on={dragSh} axis="x" label={`sh ${tn_sh.toFixed(2)}`} />
      </svg>

      <div className="xv-readout-grid xv-tone-readouts">
        <Readout label="Grey" value={tn_lg.toFixed(1)} accent={isModified(tn_lg, g.baseConfig.tn_lg) ? 'var(--xv-primary)' : undefined}
          scrub={{ get: () => g.effective('tn_lg'), set: (v) => g.setDrt('tn_lg', +v.toFixed(1)), min: lgMin, max: lgMax }} />
        <Readout label="Shadows" value={tn_toe.toFixed(3)} accent={isModified(tn_toe, g.baseConfig.tn_toe) ? 'var(--xv-primary)' : undefined}
          scrub={{ get: () => g.effective('tn_toe'), set: (v) => g.setDrt('tn_toe', +v.toFixed(4)), min: 0, max: 0.1 }} />
        <Readout label="Highlights" value={tn_sh.toFixed(2)} accent={isModified(tn_sh, g.baseConfig.tn_sh) ? 'var(--xv-primary)' : undefined}
          scrub={{ get: () => g.effective('tn_sh'), set: (v) => g.setDrt('tn_sh', +v.toFixed(2)), min: 0.15, max: 0.95 }} />
      </div>

      <div className="xv-tone-sliders">
        <Slider label="Contrast" min={conMin} max={conMax} step={0.01}
          value={tn_con} defaultValue={g.baseConfig.tn_con}
          onChange={(v) => g.setDrt('tn_con', v)} />
        <Slider label="Local contrast" min={0} max={2} step={0.01}
          value={cfg.tn_lcon_enable ? cfg.tn_lcon : 0}
          defaultValue={g.baseConfig.tn_lcon_enable ? g.baseConfig.tn_lcon : 0}
          onChange={(v) => {
            g.setDrt('tn_lcon_enable', v !== 0);
            g.setDrt('tn_lcon', v);
          }} />
      </div>
    </div>
  );
}
