import { useAppStore } from '@/store';
import { PhotoStage } from './PhotoStage';
import { TopBar } from './TopBar';
import { Filmstrip } from './Filmstrip';
import { ActionHud } from './ActionHud';
import { EmptyState } from './EmptyState';
import { ToolRail } from './ToolRail';
import { PanelHost } from './PanelHost';
import { HistogramHud } from './HistogramHud';
import './HudRoot.css';

export function HudRoot() {
  const hasFiles = useAppStore((s) => s.files.length > 0);
  const initialized = useAppStore((s) => s.initialized);
  const initError = useAppStore((s) => s.initError);

  // With files: full shell. PhotoStage only mounts here, so its "Select a file"
  // placeholder can never bleed through the empty state.
  if (hasFiles) {
    return (
      <div className="xv-hud-root">
        <PhotoStage />
        <div className="xv-hud-overlay">
          <TopBar />
          <Filmstrip />
          <ActionHud />
          <ToolRail />
          <PanelHost />
          <HistogramHud />
        </div>
      </div>
    );
  }

  // No files: the status pill is always shown (loading / backend / error). The
  // drop zone appears only once startup has settled (initialized or errored), so
  // a session restore doesn't flash the empty state before its files load.
  return (
    <div className="xv-hud-root">
      <div className="xv-hud-overlay">
        <TopBar />
      </div>
      {(initialized || initError) && <EmptyState />}
    </div>
  );
}
