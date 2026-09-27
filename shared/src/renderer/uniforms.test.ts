import { describe, expect, it } from 'vitest';
import { applyOpenDrtUniforms, UNIFORM_FLOATS, U_CWP_C0, U_ODRT_RS, U_P3_DSP_C0 } from './uniforms';
import { configFromPreset, configWithOverrides, computeTonescaleParams } from './grading/opendrt-params';
import { computeCwpAdaptMatrix } from './color-matrices';

describe('OpenDRT output uniforms', () => {
  function pack(gamut: 'rec709' | 'p3' | 'rec2020', warmth: number) {
    const cfg = configWithOverrides(configFromPreset('default'), { cwp: warmth, cwp_rng: 0.5, rs_sa: 1 });
    const data = new Float32Array(UNIFORM_FLOATS);
    applyOpenDrtUniforms(data, computeTonescaleParams(cfg), cfg, gamut);
    return data;
  }
  it('uses P3 adaptation for Rec.2020 and defers only its output conversion', () => {
    const rec709 = pack('rec709', 1), p3 = pack('p3', 1), rec2020 = pack('rec2020', 1);
    expect(rec2020[U_ODRT_RS + 3]).toBe(1);
    expect(p3[U_ODRT_RS + 3]).toBe(0);
    expect(rec709[U_ODRT_RS + 3]).toBe(0);
    expect(rec2020.slice(U_CWP_C0)).toEqual(p3.slice(U_CWP_C0));
    expect(rec2020.slice(U_CWP_C0)).not.toEqual(rec709.slice(U_CWP_C0));
    expect(rec2020[U_P3_DSP_C0]).toBeCloseTo(0.7538330344);
  });
  it('interpolates warmth continuously between D65 and D50', () => {
    const cool = pack('rec709', 0), half = pack('rec709', 0.5), warm = pack('rec709', 1);
    for (let i = U_CWP_C0; i < UNIFORM_FLOATS; i++) {
      expect(half[i]).toBeCloseTo((cool[i] + warm[i]) / 2, 6);
    }
    expect(half).not.toEqual(warm);
    expect(computeCwpAdaptMatrix(true, 0.01)).not.toEqual(computeCwpAdaptMatrix(true, 1));
  });
  it('protects the GPU even when a caller supplies unsanitized overrides', () => {
    const cfg = { ...configWithOverrides(configFromPreset('default'), {}), rs_sa: 1 };
    const data = new Float32Array(UNIFORM_FLOATS);
    applyOpenDrtUniforms(data, computeTonescaleParams(cfg), cfg, 'rec709');
    expect(data[U_ODRT_RS]).toBeCloseTo(0.6);
  });
});
