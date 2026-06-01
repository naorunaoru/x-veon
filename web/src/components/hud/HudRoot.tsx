import { useAppStore } from '@/store';
import { PhotoStage } from './PhotoStage';
import { TopBar } from './TopBar';
import { Filmstrip } from './Filmstrip';
import { ActionHud } from './ActionHud';
import { EmptyState } from './EmptyState';
import './HudRoot.css';

export function HudRoot() {
  const hasFiles = useAppStore((s) => s.files.length > 0);

  return (
    <div className="xv-hud-root">
      <PhotoStage />
      {hasFiles ? (
        <div className="xv-hud-overlay">
          <TopBar />
          <Filmstrip />
          <ActionHud />
        </div>
      ) : (
        <EmptyState />
      )}
    </div>
  );
}
