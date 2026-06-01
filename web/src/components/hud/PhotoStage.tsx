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

  const isReprocessing = selectedFile.status === 'processing' && !!selectedFile.result;

  return (
    <div className="xv-stage">
      {selectedFile.result ? (
        <>
          <OutputCanvas key={selectedFile.id} fileId={selectedFile.id} result={selectedFile.result} />
          {isReprocessing && (
            <div className="xv-stage__overlay"><span className="xv-stage__spinner" /></div>
          )}
        </>
      ) : (
        <div className="xv-stage__center">
          {selectedFile.status === 'error'
            ? <span className="xv-stage__error">{selectedFile.error}</span>
            : <span className="xv-stage__spinner" />}
        </div>
      )}
    </div>
  );
}
