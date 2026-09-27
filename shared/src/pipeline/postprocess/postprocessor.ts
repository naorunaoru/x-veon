import { XYZ_TO_SRGB } from '../constants';

function invert3x3(m: Float32Array): Float32Array {
  const [a, b, c, d, e, f, g, h, i] = m;
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  const inv = 1 / det;
  return new Float32Array([
    (e * i - f * h) * inv, (c * h - b * i) * inv, (b * f - c * e) * inv,
    (f * g - d * i) * inv, (a * i - c * g) * inv, (c * d - a * f) * inv,
    (d * h - e * g) * inv, (b * g - a * h) * inv, (a * e - b * d) * inv,
  ]);
}

function mul3x3(a: Float32Array, b: Float32Array): Float32Array {
  const r = new Float32Array(9);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      r[i * 3 + j] =
        a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
    }
  }
  return r;
}

export function buildColorMatrix(xyzToCam: Float32Array): Float32Array {
  const srgbToXyz = invert3x3(new Float32Array(XYZ_TO_SRGB));
  const srgbToCam = mul3x3(new Float32Array(xyzToCam), srgbToXyz);

  for (let i = 0; i < 3; i++) {
    const sum = srgbToCam[i * 3] + srgbToCam[i * 3 + 1] + srgbToCam[i * 3 + 2];
    srgbToCam[i * 3] /= sum;
    srgbToCam[i * 3 + 1] /= sum;
    srgbToCam[i * 3 + 2] /= sum;
  }

  return invert3x3(srgbToCam);
}
