// Inputs for check_shader.py. Use production preset, validation and uniform packing.
import { configFromPreset, configWithOverrides, computeTonescaleParams, LOOK_PRESETS } from '../../shared/src/renderer/grading/opendrt-params';
import { applyOpenDrtUniforms, setMat3, UNIFORM_FLOATS } from '../../shared/src/renderer/uniforms';
import { SRGB_TO_P3D65 } from '../../shared/src/renderer/color-matrices';
import { EXPERT_GROUPS } from '../../shared/src/components/hud/panels/rendering/constants';
import type { LookPreset } from '../../shared/src/lib/types';
import type { OpenDrtConfig } from '../../shared/src/renderer/grading/opendrt-params';

const cases: object[] = [];
function add(name: string, overrides: Partial<OpenDrtConfig> = {}, preset: LookPreset = 'default', gamut: 'rec709' | 'p3' | 'rec2020' = 'rec709', compare = false) {
  const cfg = configWithOverrides(configFromPreset(preset), overrides);
  const data = new Float32Array(UNIFORM_FLOATS);
  applyOpenDrtUniforms(data, computeTonescaleParams(cfg), cfg, gamut);
  setMat3(data, 52, SRGB_TO_P3D65);
  data[6] = 1; // Linear output, normalized to display peak, as in export.
  cases.push({ name, cfg, gamut, compare, uniforms: Array.from(data) });
}
for (const preset of Object.keys(LOOK_PRESETS) as LookPreset[]) {
  for (const gamut of ['rec709', 'p3', 'rec2020'] as const) {
    for (const peak of [100, 1000]) add(`${preset}/${gamut}/${peak}`, { peak_luminance: peak }, preset, gamut, true);
  }
}
add('old-render-strength', { rs_sa: 1 });
add('old-zero-ranges', { pt_rng_low: 0, pt_rng_high: 0, ptm_low_st: 0, ptm_high_st: 0, hs_rgb_rng: 0 }, 'default', 'rec709', true);
add('zero-width-offset', { tn_lcon_w: 0, tn_off: 0 });
for (const cwp of [0, 0.01, 0.5, 1]) add(`warmth/${cwp}`, { cwp, cwp_rng: 0.5 });
for (const gamut of ['rec709', 'p3', 'rec2020'] as const) {
  for (const range of [0, 1]) add(`white-range/${gamut}/${range}`, { cwp: 1, cwp_rng: range }, 'default', gamut, true);
}
let seed = 904;
const rand = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
for (let i = 0; i < 1000; i++) {
  const overrides: Record<string, number | boolean> = {};
  for (const row of EXPERT_GROUPS.flatMap(g => g.rows)) {
    if (row.kind === 'toggle') overrides[row.key] = rand() > 0.5;
    else {
      const v = rand();
      overrides[row.key] = v < 0.1 ? row.min : v > 0.9 ? row.max : Math.round((row.min + rand() * (row.max - row.min)) / row.step) * row.step;
    }
  }
  add(`random/${i}`, overrides);
}
console.log(JSON.stringify(cases));
