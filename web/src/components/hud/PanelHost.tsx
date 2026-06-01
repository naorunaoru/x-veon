import { useAppStore } from '@/store';
import { ExposurePanel } from './panels/ExposurePanel';
import { WhiteBalancePanel } from './panels/WhiteBalancePanel';
import { BrilliancePanel } from './panels/BrilliancePanel';
import { LooksPanel } from './panels/LooksPanel';
import { SettingsPanel } from './panels/SettingsPanel';
import { ScopesPanel } from './panels/ScopesPanel';
import { ToneCurvePanel } from './panels/ToneCurvePanel';

/** Renders the single open floating panel (one at a time). */
export function PanelHost() {
  const openPanel = useAppStore((s) => s.openPanel);
  switch (openPanel) {
    case 'scopes': return <ScopesPanel />;
    case 'exposure': return <ExposurePanel />;
    case 'whiteBalance': return <WhiteBalancePanel />;
    case 'toneCurve': return <ToneCurvePanel />;
    case 'brilliance': return <BrilliancePanel />;
    case 'looks': return <LooksPanel />;
    case 'settings': return <SettingsPanel />;
    default: return null;
  }
}
