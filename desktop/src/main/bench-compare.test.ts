import { expect, it } from 'vitest';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
const report = (encodeMs = 85, totalMs = 130, hash = 'a'.repeat(64)) => ({
  status: 'BENCH', sample: 'bench-26mp.RAF', commit: '393d689', width: 6240, height: 4160,
  adapter: { vendor: 'apple', architecture: 'apple' },
  runs: [1, 2, 3].map(() => ({ encodeMs, totalMs, bytes: 100, sha256: hash })),
  median: { encodeMs, totalMs },
});
it.each([
  ['passes encode 8.5 with total 6.5', () => report(), 0, []],
  ['fails encode 7.9', () => report(79), 2, []],
  ['honors minimum 10', () => report(90), 2, ['--min-ratio', '10']],
  ['rejects unstable hashes', () => { const r = report(); r.runs[1].sha256 = 'c'.repeat(64); return r; }, 2, []],
  ['rejects different commits', () => ({ ...report(), commit: 'abcdef0' }), 2, []],
  ['rejects falsified median', () => ({ ...report(79), median: { encodeMs: 90, totalMs: 130 } }), 2, []],
  ['rejects empty runs', () => ({ ...report(), runs: [] }), 2, []],
  ['rejects missing encode', () => ({ ...report(), runs: report().runs.map(r => ({ ...r, encodeMs: null })) }), 2, []],
  ['rejects zero timings', () => report(0), 2, []],
  ['rejects malformed hash', () => report(85, 130, ''), 2, []],
  ['rejects invalid dimensions', () => ({ ...report(), width: 0 }), 2, []],
  ['rejects invalid commit', () => ({ ...report(), commit: '' }), 2, []],
  ['rejects fewer than three runs', () => ({ ...report(), runs: report().runs.slice(1) }), 2, []],
] as const)('%s', (_name, web, status, args) => {
  const dir = mkdtempSync(join(tmpdir(), 'bench-compare-'));
  try {
    writeFileSync(join(dir, 'web.json'), JSON.stringify(web()));
    // Actual desktop terminal shape; deliberately different hash across builds.
    writeFileSync(join(dir, 'desktop.json'), JSON.stringify({ status: 'BENCH', totalMs: 500, report: report(10, 20, 'b'.repeat(64)) }));
    const result = spawnSync(process.execPath, [fileURLToPath(new URL('../../scripts/bench-compare.mjs', import.meta.url)), join(dir, 'web.json'), join(dir, 'desktop.json'), ...args], { encoding: 'utf8' });
    expect(result.status, result.stderr).toBe(status);
    if (status === 0) { expect(result.stdout).toContain('8.5'); expect(result.stdout).toContain('6.5'); }
  } finally { rmSync(dir, { recursive: true, force: true }); }
});
