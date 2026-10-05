import { readFileSync, writeFileSync } from 'node:fs';
function median(values) {
  const sorted = [...values].sort((a, b) => a - b), middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}
const positive = value => typeof value === 'number' && Number.isFinite(value) && value > 0;
function load(file) {
  const raw = JSON.parse(readFileSync(file, 'utf8'));
  if (raw.status !== 'BENCH') throw Error(`${file}: expected BENCH`);
  const report = raw.report ?? raw;
  if (report.status !== 'BENCH' || report.sample !== 'bench-26mp.RAF'
      || typeof report.commit !== 'string' || !/^[a-f0-9]{7,40}$/.test(report.commit)
      || ![report.width, report.height].every(value => Number.isSafeInteger(value) && value > 0)
      || !Array.isArray(report.runs) || report.runs.length !== 3) throw Error(`${file}: malformed benchmark`);
  for (const run of report.runs) {
    if (!run || !positive(run.encodeMs) || !positive(run.totalMs)
        || !Number.isSafeInteger(run.bytes) || run.bytes <= 0 || !/^[a-f0-9]{64}$/.test(run.sha256 ?? ''))
      throw Error(`${file}: invalid measured run`);
  }
  if (new Set(report.runs.map(run => run.sha256)).size !== 1) throw Error(`${file}: unstable AVIF hashes`);
  const computed = { encodeMs: median(report.runs.map(run => run.encodeMs)), totalMs: median(report.runs.map(run => run.totalMs)) };
  if (report.median?.encodeMs !== computed.encodeMs || report.median?.totalMs !== computed.totalMs)
    throw Error(`${file}: reported median disagrees with measured runs`);
  return report;
}
try {
  const args = process.argv.slice(2), flag = args.indexOf('--min-ratio');
  let minimum = 8;
  if (flag >= 0) { minimum = Number(args[flag + 1]); args.splice(flag, 2); }
  if (!positive(minimum) || args.length < 2 || args.length > 3 || args.some(arg => arg.startsWith('--')))
    throw Error('Usage: bench-compare.mjs <web.json> <desktop.json> [out.json] [--min-ratio 8]');
  const web = load(args[0]), desktop = load(args[1]);
  for (const key of ['sample', 'commit', 'width', 'height'])
    if (web[key] !== desktop[key]) throw Error(`Benchmark ${key} differs`);
  const encodeRatio = web.median.encodeMs / desktop.median.encodeMs, totalRatio = web.median.totalMs / desktop.median.totalMs;
  if (!positive(encodeRatio) || !positive(totalRatio)) throw Error('Invalid benchmark ratios');
  const result = { status: encodeRatio >= minimum ? 'PASS' : 'FAIL', minimum, sample: web.sample, commit: web.commit,
    width: web.width, height: web.height, web: { median: web.median, sha256: web.runs[0].sha256 },
    desktop: { median: desktop.median, sha256: desktop.runs[0].sha256 }, encodeRatio, totalRatio };
  console.log(JSON.stringify(result, null, 2));
  if (args[2]) writeFileSync(args[2], JSON.stringify(result, null, 2));
  if (encodeRatio < minimum) throw Error(`Median encode ratio ${encodeRatio} is below ${minimum}`);
} catch (error) { console.error(error.message); process.exitCode = 2; }
