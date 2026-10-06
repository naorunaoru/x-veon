import path from 'node:path';
import type { DisplayReadings } from '@/host';
import { addonFile } from '../worker/native';
import { displayReadingsFrom } from '../protocol/security';

export interface DisplayReader { read(handle: Buffer | null): DisplayReadings | null }

/** Main loads the worker's addon file on first use. A missing or broken addon means no readings,
 * so the window falls back to the media-query probe (spec §10). */
export function loadDisplayReader(load: (file: string) => unknown = file => require(file), dir = path.resolve(__dirname, '../../native')): DisplayReader {
  let addon: { displayReadings?: (handle?: Buffer) => unknown } | null | undefined;
  return {
    read(handle) {
      if (addon === undefined) {
        try { addon = load(path.join(dir, addonFile())) as typeof addon; }
        catch (error) { console.warn('Display readings unavailable:', error); addon = null; }
      }
      if (typeof addon?.displayReadings !== 'function') return null;
      try { return displayReadingsFrom(addon.displayReadings(handle ?? undefined)); }
      catch (error) { console.warn('Display readings failed:', error); return null; }
    },
  };
}
