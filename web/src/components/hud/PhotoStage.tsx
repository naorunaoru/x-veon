import { ImageOff } from 'lucide-react';
import { useAppStore } from '@/app/store';
import { OutputCanvas } from '@/components/OutputCanvas';
import './PhotoStage.css';

export function PhotoStage() {
  const selectedFile = useAppStore((s) => s.files.find((f) => f.id === s.selectedFileId));

  if (!selectedFile) {
    return (
      <div className="xv-stage">
        <div className="xv-stage__center">Select a file</div>
      </div>
    );
  }

  // One loading indicator for the whole "being worked on" window — covers fresh
  // processing, restore re-queue, and reprocess-on-switch alike. Shown over the
  // canvas (dimming it) when a prior result exists, or over the black stage.
  const loading = selectedFile.status === 'queued' || selectedFile.status === 'processing';
  // Show the error whenever the file errored — don't gate on a missing result, or a
  // stale/garbage result would leave a silent black canvas with no error shown.
  const showError = selectedFile.status === 'error';

  return (
    <div className="xv-stage">
      {selectedFile.result && (
        <OutputCanvas key={selectedFile.id} fileId={selectedFile.id} result={selectedFile.result} />
      )}
      {showError && (
        <div className="xv-stage__center">
          <div className="xv-stage__error xv-glass-heavy">
            <ImageOff className="xv-stage__error-icon" size={30} strokeWidth={1.5} />
            <span className="xv-stage__error-title">Couldn't open this photo</span>
            <span className="xv-stage__error-detail">{selectedFile.error}</span>
            <span className="xv-stage__error-file">{selectedFile.originalName}</span>
          </div>
        </div>
      )}
      {loading && (
        <div className="xv-stage__overlay"><span className="xv-stage__spinner" /></div>
      )}
    </div>
  );
}
