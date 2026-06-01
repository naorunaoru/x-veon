import { useAppStore } from '@/store';
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
  const showError = selectedFile.status === 'error' && !selectedFile.result;

  return (
    <div className="xv-stage">
      {selectedFile.result && (
        <OutputCanvas key={selectedFile.id} fileId={selectedFile.id} result={selectedFile.result} />
      )}
      {showError && (
        <div className="xv-stage__center">
          <span className="xv-stage__error">{selectedFile.error}</span>
        </div>
      )}
      {loading && (
        <div className="xv-stage__overlay"><span className="xv-stage__spinner" /></div>
      )}
    </div>
  );
}
