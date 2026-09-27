/**
 * Release channel helpers. One build-time value (XV_CHANNEL, see vite.config.ts)
 * decides the base path, the browser-storage namespace and the build stamp.
 * Pure functions only; safe to import from vite.config.ts (Node) and from tests.
 */

export type Channel = 'stable' | 'beta' | 'dev';

export interface BuildInfo {
  channel: Channel;
  sha: string;
  date: string;
}

export const CHANNELS = ['stable', 'beta', 'dev'] as const satisfies readonly Channel[];

/** Path prefix of the GitHub Pages site. */
export const SITE_ROOT = '/x-veon/';

export function isChannel(v: unknown): v is Channel {
  return typeof v === 'string' && (CHANNELS as readonly string[]).includes(v);
}

/** Vite `base` for a channel. */
export function basePath(channel: Channel): string {
  switch (channel) {
    case 'stable': return SITE_ROOT;
    case 'beta': return `${SITE_ROOT}beta/`;
    case 'dev': return '/';
  }
}

/**
 * Browser-storage names. Deliberately different from the frozen app's
 * `xtrans-demosaic` database and its top-level `raw/` / `thumbnails/` folders,
 * so two builds on the same origin never touch each other's library.
 */
export function storageNames(channel: Channel): { dbName: string; opfsRoot: string } {
  return { dbName: `xveon-${channel}`, opfsRoot: channel };
}

export function channelLabel(channel: Channel): string {
  return channel.charAt(0).toUpperCase() + channel.slice(1);
}

/** The counterpart channel a build can link to; dev builds link nowhere. */
export function otherChannelLink(
  channel: Channel,
): { channel: 'stable' | 'beta'; label: string; href: string } | null {
  if (channel === 'stable') return { channel: 'beta', label: 'beta', href: basePath('beta') };
  if (channel === 'beta') return { channel: 'stable', label: 'stable', href: basePath('stable') };
  return null;
}

/** Build stamp injected by Vite; falls back to a dev stamp under Vitest. */
export const BUILD: BuildInfo =
  typeof __XV_BUILD__ !== 'undefined' && __XV_BUILD__
    ? __XV_BUILD__
    : { channel: 'dev', sha: 'test', date: '' };
