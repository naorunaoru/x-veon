import type { ReactNode } from 'react';
import { RotateCcw, X } from 'lucide-react';
import './FloatingPanel.css';

interface FloatingPanelProps {
  title: string;
  embedded?: boolean;
  modified?: boolean;
  onReset?: () => void;
  onClose: () => void;
  children: ReactNode;
}

export function FloatingPanel({ title, embedded = false, modified = false, onReset, onClose, children }: FloatingPanelProps) {
  return (
    <div className={embedded ? "xv-adjustment-group" : "xv-panel xv-glass-heavy"}>
      <div className="xv-panel__header">
        <span className="xv-panel__title">{title}</span>
        {modified && <span className="xv-panel__dot" />}
        <div className="xv-panel__actions">
          {modified && onReset && (
            <button className="xv-panel__icon" aria-label="Reset to preset" onClick={onReset}>
              <RotateCcw size={14} />
            </button>
          )}
          {!embedded && <button className="xv-panel__icon" aria-label="Close panel" onClick={onClose}>
            <X size={14} />
          </button>}
        </div>
      </div>
      <div className="xv-panel__body">{children}</div>
    </div>
  );
}
