import type { Host } from '@/host';
import type { DesktopBridge } from '../protocol/bridge';
import { BUILD } from '@/lib/channel';
import { createLibraryWithClient } from './library';
import { createExporter } from './exporter';
import { createDisplayHost } from './display';
export function createDesktopHost(bridge: DesktopBridge): Host {
  const { library, client } = createLibraryWithClient(bridge);
  return { library, exporter: createExporter(bridge, client), display: createDisplayHost(bridge), build: BUILD, settingsDbName: 'xveon-desktop' };
}
