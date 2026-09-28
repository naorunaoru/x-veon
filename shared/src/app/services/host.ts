import type { Host } from '@/host';
let host: Host | null = null;
export function setHost(value: Host | null): void {
  host = value;
}
export function getHost(): Host {
  if (!host) throw new Error('host not installed');
  return host;
}
