export type ReleaseChannel = 'beta' | 'stable';
export interface ReleaseTag { tag: string; channel: ReleaseChannel; year: number; month: number; day: number; n: number }

// The same-day counter runs from -2 to -99, which fits every OS's version fields.
const TAG = /^(beta|stable)\/(\d{4})-(\d{2})-(\d{2})(?:-([2-9]|[1-9]\d))?$/;

/** RELEASING.md's tags: `beta/YYYY-MM-DD` or `stable/YYYY-MM-DD`, then `-2` … `-99` on the same day. */
export function parseReleaseTag(tag: string): ReleaseTag | null {
  const m = TAG.exec(tag);
  if (!m) return null;
  const [year, month, day] = [Number(m[2]), Number(m[3]), Number(m[4])];
  // Date.UTC maps years below 100 to 19xx, so this rejects those too.
  const date = new Date(Date.UTC(year, month - 1, day));
  if (date.getUTCFullYear() !== year || date.getUTCMonth() !== month - 1 || date.getUTCDate() !== day) return null;
  return { tag, channel: m[1] as ReleaseChannel, year, month, day, n: m[5] === undefined ? 1 : Number(m[5]) };
}

export function compareTags(a: ReleaseTag, b: ReleaseTag): number {
  return a.year - b.year || a.month - b.month || a.day - b.day || a.n - b.n;
}

/** macOS uses three integers for the short version and one to three for the bundle version.
 * Windows uses four parts of at most 65535, the fourth from buildNumber. */
export interface PlatformVersions { macShortVersion: string; macBundleVersion: string; windowsVersion: string; buildNumber: string }

/** A same-day stable tag keeps its display version; OS fields carry the counter. */
export function releaseVersion(t: ReleaseTag): { version: string; platform: PlatformVersions } {
  const base = `${t.year}.${t.month}.${t.day}`;
  const date = `${t.year}${String(t.month).padStart(2, '0')}${String(t.day).padStart(2, '0')}`;
  return {
    version: t.channel === 'beta' ? `${base}-beta.${t.n}` : base,
    platform: { macShortVersion: base, macBundleVersion: `${date}.${t.n}`, windowsVersion: `${base}.${t.n}`, buildNumber: String(t.n) },
  };
}

export interface Installers { mac: string; win: string }

/** electron-builder writes these names, and the update check looks for exactly them. */
export function installerNames(artifactBase: string, version: string): Installers {
  return { mac: `${artifactBase}-${version}-mac-arm64.dmg`, win: `${artifactBase}-${version}-win-x64-setup.exe` };
}

const BETA = { appId: 'io.github.naorunaoru.xveon.beta', productName: 'X-veon Beta', artifactBase: 'X-veon-Beta' };
const STABLE = { appId: 'io.github.naorunaoru.xveon', productName: 'X-veon', artifactBase: 'X-veon' };
const names = (channel: ReleaseChannel) => (channel === 'beta' ? BETA : STABLE);

export function releaseInstallers(t: ReleaseTag): Installers {
  return installerNames(names(t.channel).artifactBase, releaseVersion(t).version);
}

export interface AppIdentity {
  channel: 'beta' | 'stable' | 'dev'; appId: string; productName: string; artifactBase: string; version: string;
  tag?: string; platform?: PlatformVersions; installers: Installers;
}

/** An undefined tag makes a local build; every supplied string must be a valid release tag. */
export function appIdentity(tag: string | undefined, sha: string, dist: boolean): AppIdentity {
  if (tag === undefined) {
    const version = `0.0.0-dev.${sha}`;
    return { ...BETA, channel: dist ? 'beta' : 'dev', version, installers: installerNames(BETA.artifactBase, version) };
  }
  const parsed = parseReleaseTag(tag);
  if (!parsed) throw new Error(`Not a release tag: ${JSON.stringify(tag)}`);
  const { version, platform } = releaseVersion(parsed);
  const identity = names(parsed.channel);
  return { ...identity, channel: parsed.channel, version, tag, platform, installers: installerNames(identity.artifactBase, version) };
}
