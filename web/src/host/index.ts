import type { Host } from '@/host';
import { BUILD } from '@/lib/channel';
import { storageNames, otherChannelLink } from './channel';
import { createWebLibrary } from './library';
import { createExporter } from './exporter';
import { createDisplayHost } from './display';
import { triggerDownload } from './download';
export function createWebHost(options: { deliver?: typeof triggerDownload; reload?: () => void } = {}): Host {
  const storage = storageNames(BUILD.channel);
  return {
    library: createWebLibrary({ ...storage, reload: options.reload ?? (() => location.reload()) }),
    exporter: createExporter(options.deliver),
    display: createDisplayHost(),
    build: BUILD,
    settingsDbName: storage.dbName,
    channelLink: otherChannelLink(BUILD.channel) ?? undefined,
  };
}
