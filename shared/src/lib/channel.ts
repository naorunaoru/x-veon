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
  tag?: string;
  version?: string;
}

export const CHANNELS = ['stable', 'beta', 'dev'] as const satisfies readonly Channel[];

export function isChannel(v: unknown): v is Channel { return typeof v === 'string' && (CHANNELS as readonly string[]).includes(v); }

export function channelLabel(channel: Channel): string {
  return channel.charAt(0).toUpperCase() + channel.slice(1);
}

/** Build stamp injected by Vite; falls back to a dev stamp under Vitest. */
export const BUILD: BuildInfo =
  typeof __XV_BUILD__ !== 'undefined' && __XV_BUILD__
    ? __XV_BUILD__
    : { channel: 'dev', sha: 'test', date: '' };
