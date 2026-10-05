import type { ExportFormat } from '@/lib/types';
import type { PhotoId, ExportHost } from '@/host';
export function createExporter(): ExportHost {
  return {
    async status() {
      return { available: false, reason: 'Desktop export arrives in M3.' };
    },
    async chooseDestination(_photoId: PhotoId, _suggestedName: string, _format: ExportFormat) {
      return null;
    },
    async encode() {
      throw Error('Desktop export arrives in M3.');
    },
  };
}
