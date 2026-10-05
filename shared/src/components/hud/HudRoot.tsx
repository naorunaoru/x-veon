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
import { StartupError } from './StartupError';
import { ExportStatus } from './ExportStatus';
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
  // Hide photo controls until the selected photo is
  // actually displayable — i.e. while loading, on a decode error, or with no
  // result yet. Settings stays reachable for recovery and Clear library.
  const chromeHidden = useAppStore((s) => {
    const f = s.files.find((x) => x.id === s.selectedFileId);
    return !f || f.status !== 'done' || !f.result;
  });

  // Keep Settings mounted across empty/error transitions so Clear failures retain their notice.
  return (
    <div className="xv-hud-root">
      {hasFiles && <PhotoStage />}
      <div className={`xv-hud-overlay${chromeHidden ? ' xv-hud-overlay--folder-only' : ''}`} data-chrome-hidden={(hasFiles && chromeHidden) || undefined}>
        <TopBar folderOnly={hasFiles && chromeHidden} />
        {hasFiles && (
          <>
            <Filmstrip />
            <ActionHud />
            <div className="xv-bottom-left-hud">
              <Minimap />
              <HistogramHud />
            </div>
          </>
        )}
      </div>
      <div className="xv-hud-overlay xv-hud-settings">
        <ExportStatus />
        <ToolRail settingsOnly={chromeHidden} />
        <PanelHost settingsOnly={chromeHidden} />
      </div>
      {hasFiles ? (
        dragging && <DropSurface overlay active fileCount={fileCount} />
      ) : initError ? (
        <StartupError message={initError} />
      ) : (
        initialized && <DropSurface active={dragging} />
      )}
    </div>
  );
}
