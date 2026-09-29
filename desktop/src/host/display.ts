import type { DisplayHost } from '@/host';
import { mediaQueryHeadroom } from '@/lib/display';
export function createDisplayHost(): DisplayHost {
  return {
    async probe() {
      const info = mediaQueryHeadroom(
        matchMedia('(dynamic-range: high)').matches,
      );
      return { ...info, supported: info.headroom > 1 };
    },
  };
}
