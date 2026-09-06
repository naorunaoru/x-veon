import { useLibraryShortcuts } from '@/app/hooks/useLibraryShortcuts';
import { importFiles } from '@/app/services/library';
import { useAppStore } from '@/app/store';
import { PhotoStage } from './PhotoStage';
import { TopBar } from './TopBar';
import { Filmstrip } from './Filmstrip';
import { ActionHud } from './ActionHud';
import { DropSurface } from './DropSurface';
import { useFileDrag } from './useFileDrag';
import { ToolRail } from './ToolRail';
import { PanelHost } from './PanelHost';
import { HistogramHud } from './HistogramHud';
import { Minimap } from './Minimap';
import './HudRoot.css';

export function HudRoot() {
  useLibraryShortcuts();
  const hasFiles = useAppStore((s) => s.files.length > 0);
  const fileCount = useAppStore((s) => s.files.length);
  const initialized = useAppStore((s) => s.initialized);
  const initError = useAppStore((s) => s.initError);
  // Window-level drag detection — restores drop-to-append while a photo is open,
  // and lights the empty-state frame. Works in every state (loading / error too).
  const dragging = useFileDrag(importFiles);
  // Hide all chrome (everything but the filmstrip) until the selected photo is
  // actually displayable — i.e. while loading, on a decode error, or with no
  // result yet. The clusters fade+slide to their nearest edge (see HudRoot.css).
  const chromeHidden = useAppStore((s) => {
    const f = s.files.find((x) => x.id === s.selectedFileId);
    return !f || f.status !== 'done' || !f.result;
  });

  // With files: full shell. PhotoStage only mounts here, so its "Select a file"
  // placeholder can never bleed through the empty state.
  if (hasFiles) {
    return (
      <div className="xv-hud-root">
        <PhotoStage />
        <div className="xv-hud-overlay" data-chrome-hidden={chromeHidden || undefined}>
          <TopBar />
          <Filmstrip />
          <ActionHud />
          <ToolRail />
          <PanelHost />
          <HistogramHud />
          <Minimap />
        </div>
        {dragging && <DropSurface overlay active fileCount={fileCount} />}
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
      {(initialized || initError) && <DropSurface active={dragging} />}
    </div>
  );
}
