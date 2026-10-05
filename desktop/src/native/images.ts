/** Deterministic linear RGB stress image with smooth, sharp and specular detail. */
export function syntheticImage(width: number, height: number, seed: number, peak: number): Float32Array {
  let state = seed >>> 0;
  const random = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let n = Math.imul(state ^ (state >>> 15), 1 | state);
    n ^= n + Math.imul(n ^ (n >>> 7), 61 | n);
    return ((n ^ (n >>> 14)) >>> 0) / 4294967296;
  };
  const rectangles = Array.from({ length: 5 }, () => {
    const x = Math.floor(random() * width), y = Math.floor(random() * height);
    return { x, y, w: Math.max(1, Math.floor(random() * width / 3)), h: Math.max(1, Math.floor(random() * height / 3)), rgb: [random(), random(), random()] };
  });
  const spots = Array.from({ length: 5 }, () => ({ x: random() * width, y: random() * height, radius: Math.max(1, Math.min(width, height) * (0.01 + random() * 0.03)) }));
  const data = new Float32Array(width * height * 3);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    for (let c = 0; c < 3; c++) {
      let value = 0.05 + 0.65 * (x / Math.max(1, width - 1) + y / Math.max(1, height - 1)) / 2 + c * 0.04;
      for (const r of rectangles) if (x >= r.x && x < r.x + r.w && y >= r.y && y < r.y + r.h) value = r.rgb[c];
      value += (random() * 2 - 1) * 0.03;
      for (const s of spots) value = Math.max(value, peak * Math.exp(-((x - s.x) ** 2 + (y - s.y) ** 2) / (2 * s.radius ** 2)));
      data[(y * width + x) * 3 + c] = Math.max(0, Math.min(peak, value));
    }
  }
  return data;
}
