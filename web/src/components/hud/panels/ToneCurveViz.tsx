import type { CurvePoint } from '@/lib/grading/tonescale-curve';
import './Scopes.css';

interface ToneCurveVizProps {
  points: CurvePoint[];
  width?: number;
  height?: number;
}

export function ToneCurveViz({ points, width = 268, height = 120 }: ToneCurveVizProps) {
  const path = points
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${(p.x * width).toFixed(1)} ${((1 - p.y) * height).toFixed(1)}`)
    .join(' ');
  const grid = [0.25, 0.5, 0.75];

  return (
    <svg className="xv-curve" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Tone curve">
      <g className="xv-curve__grid">
        {grid.map((g) => <line key={`h${g}`} x1={0} y1={height * g} x2={width} y2={height * g} />)}
        {grid.map((g) => <line key={`v${g}`} x1={width * g} y1={0} x2={width * g} y2={height} />)}
      </g>
      <path className="xv-curve__ref" d={`M 0 ${height} L ${width} 0`} />
      <path className="xv-curve__line" data-testid="tone-curve-path" d={path} />
    </svg>
  );
}
