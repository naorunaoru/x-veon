import { useState } from 'react';
import { cancelExport, dismissExport, useExportJobs, type ExportJobView } from '@/app/services/export-jobs';
import { getHost } from '@/app/services/host';
import './ExportStatus.css';

function ExportRow({ job }: { job: ExportJobView }) {
  const [revealError, setRevealError] = useState<string | null>(null);
  const reveal = getHost().exporter.reveal;
  const running = job.state === 'queued' || job.state === 'rendering' || job.state === 'encoding';
  const texts = {
    queued: `Waiting to export ${job.label}`,
    rendering: `Rendering ${job.label}…`,
    encoding: `Encoding ${job.label}…`,
    done: `Exported ${job.label}`,
    failed: `Couldn't export ${job.label}: ${job.error}`,
    cancelled: `Cancelled ${job.label}`,
  };
  return (
    <div className="xv-export-status__row xv-glass">
      <p>{texts[job.state]}</p>
      {revealError && <p role="alert">{revealError}</p>}
      <div className="xv-export-status__actions">
        {running && <button type="button" className="xv-btn" aria-label={`Cancel export of ${job.label}`} onClick={() => cancelExport(job.id)}>Cancel</button>}
        {job.state === 'done' && job.destination && reveal && (
          <button type="button" className="xv-btn" onClick={async () => {
            try { await reveal.open(job.destination!); setRevealError(null); }
            catch (error) { setRevealError(error instanceof Error ? error.message : String(error)); }
          }}>{reveal.label}</button>
        )}
        {(job.state === 'done' || job.state === 'failed') && <button type="button" className="xv-btn" onClick={() => dismissExport(job.id)}>Dismiss</button>}
      </div>
    </div>
  );
}
export function ExportStatus() {
  const jobs = useExportJobs((state) => state.jobs);
  return <div className="xv-export-status" role="status" aria-live="polite">{jobs.map((job) => <ExportRow key={job.id} job={job} />)}</div>;
}
