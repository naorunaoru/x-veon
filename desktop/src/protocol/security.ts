export function assetName(
  raw: string,
  method: string,
  files: ReadonlySet<string>,
): string | null {
  if (method !== 'GET' || /(?:%2e|%2f|%5c|\\|\/\.\.?\/)/i.test(raw))
    return null;
  try {
    const url = new URL(raw);
    if (
      url.protocol !== 'app:' ||
      url.host !== 'bundle' ||
      url.username ||
      url.password
    )
      return null;
    const name = decodeURIComponent(url.pathname.slice(1)) || 'index.html';
    return files.has(name) ? name : null;
  } catch {
    return null;
  }
}
export function acceptsSender(url: string, mainFrame: boolean): boolean {
  try {
    const u = new URL(url);
    return (
      mainFrame &&
      u.protocol === 'app:' &&
      u.host === 'bundle' &&
      u.pathname === '/'
    );
  } catch {
    return false;
  }
}
export type Request = {
  version: 1;
  kind: 'connect' | 'restart' | 'diagnostics';
};
export function isRequest(value: unknown): value is Request {
  if (!value || typeof value !== 'object') return false;
  const v = value as Request;
  return (
    v.version === 1 && ['connect', 'restart', 'diagnostics'].includes(v.kind)
  );
}
