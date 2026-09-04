import { useAppStore } from '@/app/store';
import { ExposurePanel } from './panels/ExposurePanel';
import { WhiteBalancePanel } from './panels/WhiteBalancePanel';
import { SettingsPanel } from './panels/SettingsPanel';
import { ScopesPanel } from './panels/ScopesPanel';
import { RenderingPanel } from './panels/RenderingPanel';
import { DetailPanel } from './panels/DetailPanel';

/** Renders the single open floating panel (one at a time). */
export function PanelHost() {
  const openPanel = useAppStore((s) => s.openPanel);
  switch (openPanel) {
    case 'scopes': return <ScopesPanel />;
    case 'exposure': return <ExposurePanel />;
    case 'whiteBalance': return <WhiteBalancePanel />;
    case 'advanced': return <RenderingPanel />;
    case 'detail': return <DetailPanel />;
    case 'settings': return <SettingsPanel />;
    default: return null;
  }
}
