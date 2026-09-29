import type { ExportHost } from '@/host';
export function createExporter(): ExportHost {
  return {
    async status() {
      return { available: false, reason: 'Desktop export arrives in M3.' };
    },
    async chooseDestination() {
      return null;
    },
    async encode() {
      throw Error('Desktop export arrives in M3.');
    },
  };
}
