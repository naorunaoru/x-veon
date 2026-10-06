import type { UpdateHost } from '@/host';
import type { DesktopBridge } from '../protocol/bridge';
export function createUpdateHost(bridge: Pick<DesktopBridge, 'checkForUpdate'>): UpdateHost {
  return { check: () => bridge.checkForUpdate().catch(() => null) };
}
