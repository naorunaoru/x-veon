import { ImageOff } from 'lucide-react';
import './PhotoStage.css';

/** Shown in place of the drop zone or a photo's spinner when initialisation failed. */
export function StartupError({ message }: { message: string }) {
  return (
    <div className="xv-stage__center" role="alert">
      <div className="xv-stage__error xv-glass-heavy">
        <ImageOff className="xv-stage__error-icon" size={30} strokeWidth={1.5} />
        <span className="xv-stage__error-title">X-veon couldn't start</span>
        <span className="xv-stage__error-detail">{message}</span>
        <span className="xv-stage__error-detail">
          It needs WebGPU and a working network connection for the models on first load.
        </span>
        <button type="button" className="xv-btn" onClick={() => location.reload()}>
          Reload
        </button>
      </div>
    </div>
  );
}
