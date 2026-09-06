import { useRef } from 'react';
import { FloatingPanel } from '../FloatingPanel';
import { useAppStore } from '@/app/store';
import { useHistogramCanvas } from '@/app/hooks/useHistogramCanvas';
import { ScopeControls } from '../ScopeControls';
import './Scopes.css';
import './Panels.css';

export function ScopesPanel() {
  const setOpenPanel = useAppStore((s) => s.setOpenPanel);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useHistogramCanvas(canvasRef);

  return (
    <FloatingPanel title="Scopes" onClose={() => setOpenPanel(null)}>
      <div className="xv-pgroup">
        <span className="xv-pgroup__title">Histogram</span>
        <canvas ref={canvasRef} width={464} height={180} className="xv-scope-canvas" />
        <ScopeControls />
      </div>
    </FloatingPanel>
  );
}
