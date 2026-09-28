import type { Channel } from '../../../shared/src/lib/channel';
export { BUILD, channelLabel, CHANNELS, isChannel } from '../../../shared/src/lib/channel';
/** Path prefix of the GitHub Pages site. */
export const SITE_ROOT = '/x-veon/';

/** Vite `base` for a channel. */
export function basePath(channel: Channel): string {
  switch (channel) {
    case 'stable':
      return SITE_ROOT;
    case 'beta':
      return `${SITE_ROOT}beta/`;
    case 'dev':
      return '/';
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

/** The counterpart channel a build can link to; dev builds link nowhere. */
export function otherChannelLink(
  channel: Channel,
): { channel: 'stable' | 'beta'; label: string; href: string } | null {
  if (channel === 'stable') return { channel: 'beta', label: 'beta', href: basePath('beta') };
  if (channel === 'beta') return { channel: 'stable', label: 'stable', href: basePath('stable') };
  return null;
}
