import type { BuildInfo } from '@/lib/channel';
import type { LibraryHost } from './library';
import type { ExportHost } from './exporter';
import type { DisplayHost } from './display';
export type * from './library';
export type * from './exporter';
export type * from './display';
export interface Host {
  library: LibraryHost;
  exporter: ExportHost;
  display: DisplayHost;
  build: BuildInfo;
  settingsDbName: string;
  channelLink?: { label: string; href: string };
}
