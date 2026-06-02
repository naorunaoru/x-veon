import type { ComponentType } from 'react';
import { BarChart3, Sun, Droplet, Sliders, Focus, Settings } from 'lucide-react';
import { useAppStore } from '@/store';
import { isSectionModified, type PanelId } from '@/lib/grading/sections';
import './ToolRail.css';

interface RailItem { id: PanelId; label: string; Icon: ComponentType<{ size?: number }>; }

// Visible rail buttons in spec order (crop still hidden until its phase).
const RAIL: RailItem[] = [
  { id: 'scopes', label: 'Scopes', Icon: BarChart3 },
  { id: 'exposure', label: 'Exposure', Icon: Sun },
  { id: 'whiteBalance', label: 'White balance', Icon: Droplet },
  { id: 'advanced', label: 'Rendering', Icon: Sliders },
  { id: 'detail', label: 'Detail', Icon: Focus },
  { id: 'settings', label: 'Settings', Icon: Settings },
];

export function ToolRail() {
  const openPanel = useAppStore((s) => s.openPanel);
  const togglePanel = useAppStore((s) => s.togglePanel);
  const file = useAppStore((s) => s.files.find((f) => f.id === s.selectedFileId));
  const drt = file?.openDrtOverrides ?? {};
  const pre = file?.preProcessOverrides ?? {};

  return (
    <div className="xv-toolrail xv-glass">
      {RAIL.map(({ id, label, Icon }) => {
        const active = openPanel === id;
        const modified = isSectionModified(id, drt, pre);
        return (
          <button
            key={id}
            className={`xv-rail-btn${active ? ' is-active' : ''}`}
            aria-label={label}
            aria-pressed={active}
            onClick={() => togglePanel(id)}
          >
            <Icon size={15} />
            {modified && !active && <span className="xv-rail-btn__dot" />}
          </button>
        );
      })}
    </div>
  );
}
