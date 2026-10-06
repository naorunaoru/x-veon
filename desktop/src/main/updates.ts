import type { NewerRelease } from '@/host';
import { compareTags, parseReleaseTag, releaseInstallers, releaseVersion, type ReleaseTag } from '../release/tags';
import { releasePage } from '../protocol/security';

export const RELEASES_API = 'https://api.github.com/repos/naorunaoru/x-veon/releases';
/** Only the smoke fixture's loopback address may replace GitHub. */
export function releasesApi(override: string | undefined): string | null {
  if (!override) return RELEASES_API;
  return /^http:\/\/127\.0\.0\.1:\d{1,5}(\/[A-Za-z0-9_-]+)*$/.test(override) ? override : null;
}
export function installerFor(platform: string, arch: string): 'mac' | 'win' | null {
  if (platform === 'darwin' && arch === 'arm64') return 'mac';
  if (platform === 'win32' && arch === 'x64') return 'win';
  return null;
}
function releaseTag(value: unknown, channel: ReleaseTag['channel'], installer: 'mac' | 'win'): ReleaseTag | null {
  if (!value || typeof value !== 'object') return null;
  const r = value as Record<string, unknown>;
  const tag = r.draft !== true && typeof r.tag_name === 'string' ? parseReleaseTag(r.tag_name) : null;
  if (!tag || tag.channel !== channel || !Array.isArray(r.assets)) return null;
  const expected = releaseInstallers(tag)[installer];
  return r.assets.some(a => !!a && typeof a === 'object' && (a as { name?: unknown }).name === expected) ? tag : null;
}
/** One check per launch. Only a newer release in the build's channel and with its installer qualifies. */
export async function checkForUpdate(opts: { tag?: string; productName: string; platform: string; arch: string; api: string | null;
  fetch: (url: string, init: RequestInit) => Promise<Response>; timeoutMs?: number }): Promise<NewerRelease | null> {
  const current = opts.tag ? parseReleaseTag(opts.tag) : null;
  const installer = installerFor(opts.platform, opts.arch);
  if (!current || !installer || !opts.api) return null;
  const url = current.channel === 'stable' ? `${opts.api}/latest` : `${opts.api}?per_page=30`;
  try {
    const response = await opts.fetch(url, {
      headers: { Accept: 'application/vnd.github+json', 'User-Agent': opts.productName },
      signal: AbortSignal.timeout(opts.timeoutMs ?? 10_000),
    });
    if (!response.ok) return null;
    const body: unknown = await response.json();
    const newest = (Array.isArray(body) ? body : [body])
      .map(r => releaseTag(r, current.channel, installer))
      .filter((t): t is ReleaseTag => !!t && compareTags(t, current) > 0)
      .sort((a, b) => compareTags(b, a))[0];
    return newest ? { name: opts.productName, version: releaseVersion(newest).version, url: releasePage(newest) } : null;
  } catch { return null; }
}
