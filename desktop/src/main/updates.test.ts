import { describe, expect, it, vi } from 'vitest';
import { checkForUpdate, installerFor, releasesApi, RELEASES_API } from './updates';

const asset = (name: string) => ({ name });
const beta = [
  { tag_name: 'stable/2026-10-09', assets: [asset('X-veon-2026.10.9-mac-arm64.dmg')] },
  { tag_name: 'beta/2026-10-07', html_url: 'https://example.com/elsewhere', assets: [asset('X-veon-Beta-2026.10.7-beta.1-mac-arm64.dmg')] },
  { tag_name: 'beta/2026-10-08', assets: [asset('X-veon-Beta-2026.10.8-beta.1-win-x64-setup.exe')] },
  { tag_name: 'beta/2026-10-09', assets: [asset('X-veon-Beta-2026.10.9-beta.1-mac-arm64.dmg.blockmap'), asset('Other-2026.10.9-mac-arm64.dmg')] },
  { tag_name: 'beta/2026-10-10', draft: true, assets: [asset('X-veon-Beta-2026.10.10-beta.1-mac-arm64.dmg')] },
  { tag_name: 123, assets: [asset('X-veon-Beta-2026.10.11-beta.1-mac-arm64.dmg')] },
];
const response = (body: unknown, ok = true) => ({ ok, status: ok ? 200 : 403, json: async () => body }) as Response;
const fetch = vi.fn(async (_url: string, _init: RequestInit) => response(beta));
const opts = { tag: 'beta/2026-10-06', productName: 'X-veon Beta', platform: 'darwin', arch: 'arm64', api: RELEASES_API, fetch };

describe('update source', () => {
  it('accepts only the fixed API or a bare loopback fixture path', () => {
    expect(releasesApi(undefined)).toBe(RELEASES_API);
    expect(releasesApi('')).toBe(RELEASES_API);
    expect(releasesApi('http://127.0.0.1:4321/releases')).toBe('http://127.0.0.1:4321/releases');
    for (const value of ['https://api.example.com/releases', 'http://localhost:4321/releases', 'http://127.0.0.1:4321/releases?x=1', 'http://127.0.0.1.example.com/releases']) expect(releasesApi(value)).toBeNull();
  });
  it('supports exactly the two packaged targets and never asks without a valid build', async () => {
    expect(installerFor('darwin', 'arm64')).toBe('mac');
    expect(installerFor('win32', 'x64')).toBe('win');
    expect(installerFor('darwin', 'x64')).toBeNull();
    for (const change of [{ tag: undefined }, { platform: 'linux' }, { arch: 'x64' }, { api: null }]) {
      fetch.mockClear();
      await expect(checkForUpdate({ ...opts, ...change })).resolves.toBeNull();
      expect(fetch).not.toHaveBeenCalled();
    }
  });
  it('finds the newest same-channel beta with the exact local installer and constructs its URL', async () => {
    fetch.mockClear();
    await expect(checkForUpdate(opts)).resolves.toEqual({ name: 'X-veon Beta', version: '2026.10.7-beta.1', url: 'https://github.com/naorunaoru/x-veon/releases/tag/beta/2026-10-07' });
    expect(fetch).toHaveBeenCalledOnce();
    expect(fetch.mock.calls[0][0]).toBe(`${RELEASES_API}?per_page=30`);
    const init = fetch.mock.calls[0][1] as RequestInit;
    expect(init.headers).toMatchObject({ Accept: 'application/vnd.github+json', 'User-Agent': 'X-veon Beta' });
    expect(init.signal).toBeInstanceOf(AbortSignal);
    await expect(checkForUpdate({ ...opts, platform: 'win32', arch: 'x64' })).resolves.toEqual({ name: 'X-veon Beta', version: '2026.10.8-beta.1', url: 'https://github.com/naorunaoru/x-veon/releases/tag/beta/2026-10-08' });
  });
  it('ignores equal and older releases and uses the fixture address', async () => {
    await expect(checkForUpdate({ ...opts, tag: 'beta/2026-10-08', fetch: async () => response(beta.slice(0, 2)) })).resolves.toBeNull();
    fetch.mockClear();
    await checkForUpdate({ ...opts, api: 'http://127.0.0.1:4321/releases' });
    expect(fetch.mock.calls[0][0]).toBe('http://127.0.0.1:4321/releases?per_page=30');
  });
  it('honors stable Latest, including rollback and a same-day counter', async () => {
    const stable = { ...opts, tag: 'stable/2026-10-06', productName: 'X-veon' };
    const old = { tag_name: 'stable/2026-10-05', assets: [asset('X-veon-2026.10.5-mac-arm64.dmg')] };
    await expect(checkForUpdate({ ...stable, fetch: async () => response(old) })).resolves.toBeNull();
    fetch.mockClear();
    fetch.mockResolvedValueOnce(response({ tag_name: 'stable/2026-10-06-2', assets: [asset('X-veon-2026.10.6-mac-arm64.dmg')] }));
    await expect(checkForUpdate(stable)).resolves.toEqual({ name: 'X-veon', version: '2026.10.6', url: 'https://github.com/naorunaoru/x-veon/releases/tag/stable/2026-10-06-2' });
    expect(fetch.mock.calls[0][0]).toBe(`${RELEASES_API}/latest`);
  });
  it('returns null for fetch, status, JSON, and shape failures', async () => {
    for (const failure of [async () => { throw new Error('network'); }, async () => response(null, false), async () => ({ ok: true, json: async () => { throw new Error('body'); } }) as unknown as Response, async () => response({ message: 'x' }), async () => response([1, 'a', null])]) {
      await expect(checkForUpdate({ ...opts, fetch: failure })).resolves.toBeNull();
    }
  });
  it.each(['headers', 'body'])('aborts stalled %s', async phase => {
    const stalled = (_url: string, init: RequestInit) => new Promise<never>((_resolve, reject) => init.signal!.addEventListener('abort', () => reject(init.signal!.reason)));
    const request = phase === 'headers' ? stalled : async (_url: string, init: RequestInit) => ({ ok: true, json: () => stalled('', init) }) as unknown as Response;
    await expect(checkForUpdate({ ...opts, fetch: request, timeoutMs: 20 })).resolves.toBeNull();
  }, 1_000);
});
