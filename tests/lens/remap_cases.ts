// Inputs for check_remap.py. Use the production table builder, uniform packing and CPU twin of the shader's remap.
import { buildLensTable, lensSourceUv } from '../../shared/src/renderer/lens-table';
import { applyLensUniforms, UNIFORM_FLOATS } from '../../shared/src/renderer/uniforms';
import { NO_CURVES } from '../../shared/src/lib/lens/curves';
import type { LensCurves } from '../../shared/src/lib/lens/curves';
import type { LensFrame } from '../../shared/src/renderer/lens-table';

const ALL = { distortion: true, ca: true, vignetting: true };
const FRAME: LensFrame = { width: 3000, height: 2000 };

// 64 uvs on an 8 × 8 grid inside (0, 1), then the four corners.
const uvs: [number, number][] = [];
for (let j = 0; j < 8; j++) for (let i = 0; i < 8; i++) uvs.push([(i + 0.5) / 8, (j + 0.5) / 8]);
uvs.push([0, 0], [1, 0], [0, 1], [1, 1]);

const cases: object[] = [];
function add(name: string, curves: Partial<LensCurves>, frame: LensFrame) {
  const remap = buildLensTable({ ...NO_CURVES, ...curves }, ALL, frame);
  if (!remap) throw new Error(`${name}: no remap`);
  const uniforms = new Float32Array(UNIFORM_FLOATS);
  applyLensUniforms(uniforms, remap);
  const expected = uvs.map((uv) => lensSourceUv(remap, uv));
  cases.push({ name, uniforms: Array.from(uniforms), table: Array.from(remap.table), uvs, expected });
}

add('barrel + CA + vignetting, off-centre', {
  distortion: (rho) => 1 - 0.1 * rho * rho,
  caRed: (r) => 1 + 0.002 * r,
  caBlue: (r) => 1 - 0.002 * r,
  vignetting: (r) => 1 + r,
}, { ...FRAME, centre: [1560, 1030] });
add('pincushion', { distortion: (rho) => 1 + 0.08 * rho * rho }, FRAME);
add('vignetting', { vignetting: (r) => 1 + 0.5 * r * r }, FRAME);
console.log(JSON.stringify(cases));
