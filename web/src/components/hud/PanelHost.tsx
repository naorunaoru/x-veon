import { useAppStore } from '@/store';
import { ExposurePanel } from './panels/ExposurePanel';
import { WhiteBalancePanel } from './panels/WhiteBalancePanel';
import { BrilliancePanel } from './panels/BrilliancePanel';
import { LooksPanel } from './panels/LooksPanel';
import { SettingsPanel } from './panels/SettingsPanel';

/** Renders the single open floating panel (one at a time). */
export function PanelHost() {
  const openPanel = useAppStore((s) => s.openPanel);
  switch (openPanel) {
    case 'exposure': return <ExposurePanel />;
    case 'whiteBalance': return <WhiteBalancePanel />;
    case 'brilliance': return <BrilliancePanel />;
    case 'looks': return <LooksPanel />;
    case 'settings': return <SettingsPanel />;
    default: return null;
  }
}
