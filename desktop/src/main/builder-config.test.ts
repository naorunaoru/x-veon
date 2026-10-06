import { afterEach, describe, expect, it } from 'vitest';
import { createRequire } from 'node:module';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { appIdentity } from '../release/tags';

const require = createRequire(import.meta.url);
const configPath = require.resolve('../../electron-builder.config.cjs');
const dirs: string[] = [];
const previous = process.env.XV_RELEASE_FILE;

function configFor(tag: string | undefined) {
  const dir = mkdtempSync(path.join(tmpdir(), 'xv-builder-'));
  dirs.push(dir);
  const file = path.join(dir, 'release.json');
  writeFileSync(file, JSON.stringify(appIdentity(tag, 'abc1234', true)));
  process.env.XV_RELEASE_FILE = file;
  delete require.cache[configPath];
  return require(configPath);
}

afterEach(() => {
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
  if (previous === undefined) delete process.env.XV_RELEASE_FILE;
  else process.env.XV_RELEASE_FILE = previous;
  delete require.cache[configPath];
});

describe('electron-builder release config', () => {
  it('uses tagged stable identity and OS versions', () => {
    const config = configFor('stable/2026-10-01-2');
    expect(config).toMatchObject({
      appId: 'io.github.naorunaoru.xveon', productName: 'X-veon', extraMetadata: { version: '2026.10.1' },
      buildVersion: '2026.10.1.2', buildNumber: '2',
      mac: { bundleShortVersion: '2026.10.1', bundleVersion: '20261001.2', artifactName: 'X-veon-2026.10.1-mac-arm64.dmg' },
      nsis: { artifactName: 'X-veon-2026.10.1-win-x64-setup.exe' },
    });
  });

  it('omits release-only OS versions for an untagged build', () => {
    const config = configFor(undefined);
    expect(config).not.toHaveProperty('buildVersion');
    expect(config).not.toHaveProperty('buildNumber');
    expect(config.mac).not.toHaveProperty('bundleShortVersion');
    expect(config.mac).not.toHaveProperty('bundleVersion');
    expect(config.mac.artifactName).toBe('X-veon-Beta-0.0.0-dev.abc1234-mac-arm64.dmg');
  });

  it('explains how to create a missing identity file', () => {
    process.env.XV_RELEASE_FILE = path.join(tmpdir(), 'xv-missing-release-identity.json');
    delete require.cache[configPath];
    expect(() => require(configPath)).toThrow('npm run dist');
  });
});
