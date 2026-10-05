vi.stubGlobal('navigator', { platform: 'MacIntel' });
import { expect, it, vi } from 'vitest';
import type { DesktopBridge } from '../protocol/bridge';
const request = vi.hoisted(() => vi.fn(async () => ({ availability: { available: true } })));
vi.mock('./port', () => ({ createWorkerClient: () => ({ request }) }));
const bridge = { chooseExportDestination: vi.fn(async () => ({ token: 'golden' })) } as unknown as DesktopBridge;
import { createGoldenHost } from './golden-fixtures';
it('keeps fixtures and edits in memory without exposing delete or clear', async () => {
  const fixture = createGoldenHost(bridge),
    host = fixture.host.library;
  expect(host.remove).toBeUndefined();
  expect(host.clear).toBeUndefined();
  await expect(host.addFiles([new File(['raw'], 'other.raf')])).rejects.toThrow(
    'two sample',
  );
  const p = (await host.addFiles([new File(['raw'], 'DSCF3332.RAF')]))!
    .photos[0];
  expect(new TextDecoder().decode(await host.readRaw(p.id))).toBe('raw');
  await host.save(
    p.id,
    { ...p.edit, preProcessOverrides: { exposure: 1 } },
    p.facts,
  );
  expect((await host.load()).photos[0].edit.preProcessOverrides.exposure).toBe(
    1,
  );
  fixture.releaseFixture(p.id);
  expect((await host.load()).photos).toEqual([]);
  expect((await createGoldenHost(bridge).host.library.load()).photos).toEqual([]);
  expect(p.id).toMatch(/^[A-Za-z0-9_-]{22}$/);
  expect(await fixture.host.exporter.status()).toEqual({ available: true });
  expect(await fixture.host.exporter.chooseDestination(p.id, 'x.avif', 'avif')).toEqual({ token: 'golden' });
  expect(bridge.chooseExportDestination).toHaveBeenCalledWith(p.id, 'avif');
});

it('isolates golden settings from production', () => { expect(createGoldenHost(bridge).host.settingsDbName).toBe('xveon-desktop-golden'); });
