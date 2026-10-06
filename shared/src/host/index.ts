import type { BuildInfo } from '@/lib/channel';
import type { LibraryHost } from './library';
import type { ExportHost } from './exporter';
import type { DisplayHost } from './display';
import type { UpdateHost } from './updates';
export type * from './library';
export type * from './exporter';
export type * from './display';
export type * from './updates';
export interface Host {
  library: LibraryHost;
  exporter: ExportHost;
  display: DisplayHost;
  /** desktop: a newer release in this build's channel (spec §9) */
  updates?: UpdateHost;
  build: BuildInfo;
  settingsDbName: string;
  channelLink?: { label: string; href: string };
}
