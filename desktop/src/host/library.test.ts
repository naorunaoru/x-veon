import { expect, it } from 'vitest';
import { createFixtureLibrary } from './library';
import { createExporter } from './exporter';
it('keeps fixtures and edits in memory without exposing delete or clear', async () => {
  const fixture = createFixtureLibrary(),
    host = fixture.library;
  expect(host.remove).toBeUndefined();
  expect(host.clear).toBeUndefined();
  await expect(host.addFiles([new File(['raw'], 'other.raf')])).rejects.toThrow(
    'two sample',
  );
  const p = (await host.addFiles([new File(['raw'], 'DSCF3332.RAF')]))
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
  expect((await createFixtureLibrary().library.load()).photos).toEqual([]);
  expect(await createExporter().status()).toMatchObject({ available: false });
});
