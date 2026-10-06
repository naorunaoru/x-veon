import { execFileSync, spawnSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';

const script = fileURLToPath(new URL('../../scripts/resolve-release.mjs', import.meta.url));
let repo = '', before = '', after = '';
// The author's global signing settings must not reach these throwaway commits and tags.
const git = (...args: string[]) => execFileSync('git', ['-c', 'commit.gpgsign=false', '-c', 'tag.gpgsign=false', ...args], { cwd: repo, encoding: 'utf8' }).trim();

beforeAll(() => {
  repo = mkdtempSync(path.join(tmpdir(), 'resolve-release-'));
  git('init', '-q');
  git('config', 'user.name', 'Test'); git('config', 'user.email', 'test@example.com');
  writeFileSync(path.join(repo, 'README'), 'before M4\n');
  git('add', 'README'); git('commit', '-qm', 'before M4');
  before = git('rev-parse', 'HEAD');
  mkdirSync(path.join(repo, 'desktop/src/release'), { recursive: true });
  writeFileSync(path.join(repo, 'desktop/src/release/tags.ts'), '// M4\n');
  git('add', 'desktop'); git('commit', '-qm', 'M4');
  after = git('rev-parse', 'HEAD');
  for (const [tag, commit] of [['beta/2026-10-06', after], ['stable/2026-10-06-2', after], ['beta/2026-02-30', after], ['beta/2026-09-25-2', before]]) {
    git('tag', '-a', tag, '-m', tag, commit);
  }
  git('tag', 'beta/2026-10-07', after); // lightweight
});
afterAll(() => rmSync(repo, { recursive: true, force: true }));

function run(env: { event: 'push' | 'workflow_dispatch'; ref: string; tag?: string; identity?: string; dryRun?: boolean }) {
  const output = path.join(repo, '.github-output');
  writeFileSync(output, '');
  // Every variable the script reads is set, so a CI job's own GITHUB_* values can't leak in.
  const result = spawnSync(process.execPath, [script], {
    cwd: repo, encoding: 'utf8',
    env: { ...process.env, GITHUB_OUTPUT: output, GITHUB_EVENT_NAME: env.event, GITHUB_REF_NAME: env.ref, GITHUB_SHA: after,
      INPUT_TAG: env.tag ?? '', INPUT_IDENTITY: env.identity ?? '', DRY_RUN: String(env.dryRun ?? false) },
  });
  const outputs = Object.fromEntries(readFileSync(output, 'utf8').split('\n').filter(Boolean)
    .map(line => [line.slice(0, line.indexOf('=')), line.slice(line.indexOf('=') + 1)]));
  return { status: result.status, stdout: result.stdout, outputs };
}
const push = (tag: string) => run({ event: 'push', ref: tag });
const dispatch = (inputs: { tag?: string; identity?: string; dryRun?: boolean }) => run({ event: 'workflow_dispatch', ref: 'codex/desktop-m4', ...inputs });

describe('resolve-release.mjs', () => {
  it('builds and publishes the commit of a pushed annotated beta tag', () => {
    expect(push('beta/2026-10-06')).toMatchObject({ status: 0, outputs: { tag: 'beta/2026-10-06', identity: 'beta/2026-10-06', sha: after, build: 'true', publish: 'true' } });
  });
  it.each([
    ['beta/2026-02-30', 'not a release tag'], ['beta/2026-09-27-1', 'not a release tag'], ['beta/2026-13-01', 'not a release tag'],
    ['beta/2026-10-06-100', 'not a release tag'], ['beta/2026-10-07', 'existing annotated tag'],
  ])('fails a pushed %s before any build', (tag, reason) => {
    const result = push(tag);
    expect(result.status).toBe(1);
    expect(result.stdout).toContain('::error::');
    expect(result.stdout).toContain(tag);
    expect(result.stdout).toContain(reason);
    expect(result.outputs).toEqual({});
  });
  it('publishes a dispatched stable tag, unless the dispatch is a dry run', () => {
    expect(dispatch({ tag: 'stable/2026-10-06-2' })).toMatchObject({ status: 0, outputs: { tag: 'stable/2026-10-06-2', sha: after, build: 'true', publish: 'true' } });
    expect(dispatch({ tag: 'stable/2026-10-06-2', dryRun: true })).toMatchObject({ status: 0, outputs: { identity: 'stable/2026-10-06-2', sha: after, publish: 'false' } });
  });
  it('fails a dispatched tag that does not exist', () => {
    expect(dispatch({ tag: 'beta/2026-10-08', dryRun: true }).status).toBe(1);
  });
  it('skips a tag on a commit without the M4 desktop code, with a notice', () => {
    const result = dispatch({ tag: 'beta/2026-09-25-2', dryRun: true });
    expect(result).toMatchObject({ status: 0, outputs: { sha: before, build: 'false', publish: 'false' } });
    expect(result.stdout).toContain('::notice::');
  });
  it('builds the dispatched commit untagged, as a dry run only', () => {
    expect(dispatch({ dryRun: true })).toMatchObject({ status: 0, outputs: { tag: '', identity: '', sha: after, build: 'true', publish: 'false' } });
    expect(dispatch({}).status).toBe(1);
  });
  it('gives a dry run a release identity from a tag that needn\'t exist', () => {
    expect(dispatch({ identity: 'beta/2026-10-06-2', dryRun: true })).toMatchObject({ status: 0, outputs: { tag: '', identity: 'beta/2026-10-06-2', sha: after, build: 'true', publish: 'false' } });
    for (const inputs of [{ identity: 'beta/2026-10-06-2' }, { identity: 'beta/2026-10-06-2', tag: 'beta/2026-10-06', dryRun: true }, { identity: 'beta/2026-02-30', dryRun: true }]) {
      expect(dispatch(inputs).status, JSON.stringify(inputs)).toBe(1);
    }
  });
});
