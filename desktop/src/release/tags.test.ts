import { describe, expect, it } from 'vitest';
import { appIdentity, compareTags, parseReleaseTag, releaseInstallers } from './tags';

describe('release tags and identity', () => {
  it('maps beta and stable tags to app identities and platform versions', () => {
    expect(appIdentity('beta/2026-09-27', 'abc1234', true)).toStrictEqual({
      channel: 'beta', appId: 'io.github.naorunaoru.xveon.beta', productName: 'X-veon Beta', artifactBase: 'X-veon-Beta',
      version: '2026.9.27-beta.1', tag: 'beta/2026-09-27',
      platform: { macShortVersion: '2026.9.27', macBundleVersion: '20260927.1', windowsVersion: '2026.9.27.1', buildNumber: '1' },
      installers: { mac: 'X-veon-Beta-2026.9.27-beta.1-mac-arm64.dmg', win: 'X-veon-Beta-2026.9.27-beta.1-win-x64-setup.exe' },
    });
    expect(appIdentity('beta/2026-09-27-2', 'x', true)).toMatchObject({
      version: '2026.9.27-beta.2', platform: { macShortVersion: '2026.9.27', macBundleVersion: '20260927.2', windowsVersion: '2026.9.27.2', buildNumber: '2' },
    });
    expect(appIdentity('stable/2026-10-01', 'x', true)).toStrictEqual({
      channel: 'stable', appId: 'io.github.naorunaoru.xveon', productName: 'X-veon', artifactBase: 'X-veon',
      version: '2026.10.1', tag: 'stable/2026-10-01',
      platform: { macShortVersion: '2026.10.1', macBundleVersion: '20261001.1', windowsVersion: '2026.10.1.1', buildNumber: '1' },
      installers: { mac: 'X-veon-2026.10.1-mac-arm64.dmg', win: 'X-veon-2026.10.1-win-x64-setup.exe' },
    });
    expect(appIdentity('stable/2026-10-01-2', 'x', true)).toMatchObject({
      version: '2026.10.1', platform: { macShortVersion: '2026.10.1', macBundleVersion: '20261001.2', windowsVersion: '2026.10.1.2', buildNumber: '2' },
    });
  });

  it('uses legal OS version formats for all supported release counters', () => {
    for (const tag of ['beta/2026-09-27', 'beta/2026-09-27-2', 'stable/2026-10-01', 'stable/2026-10-01-2', 'beta/9999-12-31-99']) {
      const v = appIdentity(tag, 'x', true).platform!;
      expect(v.macShortVersion).toMatch(/^\d+\.\d+\.\d+$/);
      expect(v.macBundleVersion).toMatch(/^\d+(\.\d+){0,2}$/);
      expect(v.windowsVersion).toMatch(/^\d+\.\d+\.\d+\.\d+$/);
      expect(v.windowsVersion.split('.').every((part) => Number(part) <= 65535)).toBe(true);
      expect(v.windowsVersion.endsWith(`.${v.buildNumber}`)).toBe(true);
    }
  });

  it('keeps untagged identities free of release-only fields', () => {
    expect(appIdentity(undefined, 'abc1234', true)).toStrictEqual({
      channel: 'beta', appId: 'io.github.naorunaoru.xveon.beta', productName: 'X-veon Beta', artifactBase: 'X-veon-Beta', version: '0.0.0-dev.abc1234',
      installers: { mac: 'X-veon-Beta-0.0.0-dev.abc1234-mac-arm64.dmg', win: 'X-veon-Beta-0.0.0-dev.abc1234-win-x64-setup.exe' },
    });
    expect(appIdentity(undefined, 'abc1234', false).channel).toBe('dev');
  });

  it('derives installer names from the same release identity', () => {
    expect(releaseInstallers(parseReleaseTag('stable/2026-10-01-2')!)).toStrictEqual(appIdentity('stable/2026-10-01-2', 'x', true).installers);
  });

  it('rejects malformed tags, impossible dates and invalid counters', () => {
    for (const tag of ['beta/2026-9-27', 'stable/2026-10-01-x', 'release/2026-10-01', 'beta/2026-09-27 ', '',
      'beta/2026-02-30', 'beta/2026-13-01', 'beta/0026-10-06', 'beta/2026-09-27-1', 'beta/2026-09-27-0',
      'beta/2026-09-27-02', 'beta/2026-09-27-100', `beta/2026-09-27-${'9'.repeat(400)}`]) {
      expect(parseReleaseTag(tag)).toBeNull();
      expect(() => appIdentity(tag, 'x', true)).toThrow(JSON.stringify(tag));
    }
    expect(parseReleaseTag('beta/2026-09-27-99')?.n).toBe(99);
  });

  it('orders by date then same-day counter regardless of channel', () => {
    const tags = ['stable/2027-01-01', 'beta/2026-09-28', 'stable/2026-09-27-2', 'beta/2026-10-01', 'beta/2026-09-27'];
    expect(tags.map((tag) => parseReleaseTag(tag)!).sort(compareTags).map((t) => t.tag)).toStrictEqual([
      'beta/2026-09-27', 'stable/2026-09-27-2', 'beta/2026-09-28', 'beta/2026-10-01', 'stable/2027-01-01',
    ]);
  });
});
