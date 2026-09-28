import type { ComponentType } from 'react';
import { Sun, Thermometer, Sliders, TriangleRight, Settings } from 'lucide-react';
import { useAppStore } from '@/app/store';
import { isSectionModified, type PanelId } from '@/renderer/grading/sections';
import { configFromPreset } from '@/renderer/grading/opendrt-params';
import './ToolRail.css';

interface RailItem { id: PanelId; label: string; Icon: ComponentType<{ size?: number }>; }

// Visible rail buttons in spec order (crop still hidden until its phase).
const RAIL: RailItem[] = [
  { id: 'exposure', label: 'Exposure', Icon: Sun },
  { id: 'advanced', label: 'Rendering', Icon: Sliders },
  { id: 'whiteBalance', label: 'White balance', Icon: Thermometer },
  { id: 'detail', label: 'Detail', Icon: TriangleRight },
  { id: 'settings', label: 'Settings', Icon: Settings },
];

export function ToolRail() {
  const openPanel = useAppStore((s) => s.openPanel);
  const setOpenPanel = useAppStore((s) => s.setOpenPanel);
  const file = useAppStore((s) => s.files.find((f) => f.id === s.selectedFileId));
  const displayHdr = useAppStore((s) => s.displayHdr);
  const headroom = useAppStore((s) => s.displayHdrHeadroom);
  const drt = file?.edit.openDrtOverrides ?? {};
  const pre = file?.edit.preProcessOverrides ?? {};
  const base = configFromPreset(file?.edit.lookPreset ?? 'default', displayHdr ? headroom : undefined);

  return (
    <div className="xv-toolrail xv-glass">
      {RAIL.map(({ id, label, Icon }) => {
        const active = openPanel === id;
        const modified = isSectionModified(id, drt, pre, base);
        return (
          <button
            key={id}
            className={`xv-rail-btn${id === 'settings' ? ' xv-rail-btn--settings' : ''}${active ? ' is-active' : ''}`}
            aria-label={label}
            title={label}
            aria-pressed={active}
            onClick={() => setOpenPanel(id === 'settings' && active ? null : id)}
          >
            <Icon size={15} />
            {modified && !active && <span className="xv-rail-btn__dot" />}
          </button>
        );
      })}
    </div>
  );
}
