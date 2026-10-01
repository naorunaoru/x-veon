import type { Host } from '@/host';
import type { DesktopBridge } from '../protocol/bridge';
import { BUILD } from '@/lib/channel';
import { createLibrary } from './library';
import { createExporter } from './exporter';
import { createDisplayHost } from './display';
export function createDesktopHost(bridge: DesktopBridge): Host {
  return { library: createLibrary(bridge), exporter: createExporter(), display: createDisplayHost(), build: BUILD, settingsDbName: 'xveon-desktop' };
}
