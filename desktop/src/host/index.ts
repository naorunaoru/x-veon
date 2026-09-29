import type { Host } from '@/host';
import { BUILD } from '@/lib/channel';
import { createFixtureLibrary } from './library';
import { createExporter } from './exporter';
import { createDisplayHost } from './display';
export function createDesktopHost() {
  const fixture = createFixtureLibrary();
  const host: Host = {
    library: fixture.library,
    exporter: createExporter(),
    display: createDisplayHost(),
    build: BUILD,
    settingsDbName: 'xveon-desktop-spike-v1',
  };
  return { host, releaseFixture: fixture.releaseFixture };
}
