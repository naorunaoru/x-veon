import { useEffect, useRef, useState } from 'react';
import type { CSSProperties } from 'react';
import { ChevronUp, X } from 'lucide-react';
import { useAppStore } from '@/app/store';
import { useExport } from '@/app/hooks/useExport';
import { getHost } from '@/app/services/host';
import {
  batchSummary, cancelBatch, dismissBatch, exportTargets, retryJob, startExport, useExportBatch, type ExportBatch,
} from '@/app/services/export-batch';
import { ExportPopover } from './ExportPopover';
import { usePopover } from './usePopover';

/** The foot's right side: the Export split button, or the batch's status while one is shown. */
export function ExportFoot() {
  const batch = useExportBatch((s) => s.batch);
  if (!batch) return <ExportButton />;
  if (batch.status === 'resolving' || batch.status === 'running') return <RunningStatus batch={batch} />;
  if (batch.status === 'done') return <DoneStatus batch={batch} />;
  return <FailedStatus batch={batch} />;
}

function ExportButton() {
  const selectedCount = useAppStore((s) => s.selectedIds.length);
  const hasTarget = useAppStore((s) => exportTargets(s).length > 0);
  const initialized = useAppStore((s) => s.initialized);
  const { exportAvailable, unavailableReason } = useExport();
  const { open, close, triggerProps, popoverProps } = usePopover();
  const disabled = !exportAvailable || !initialized || !hasTarget;
  return (
    <div className="xv-column__export-group">
      <button
        type="button"
        className="xv-column__export"
        disabled={disabled}
        title={unavailableReason ?? undefined}
        onClick={() => startExport(exportTargets(useAppStore.getState()))}
      >
        {selectedCount > 1 ? `Export ${selectedCount}` : 'Export'}
      </button>
      <button type="button" className="xv-column__export-more" aria-label="Export settings" {...triggerProps}>
        <ChevronUp size={10} />
      </button>
      {open && <ExportPopover popoverProps={popoverProps} onClose={close} exportDisabled={disabled} />}
    </div>
  );
}

function RunningStatus({ batch }: { batch: ExportBatch }) {
  const { total, completed } = batchSummary(batch);
  return (
    <div className="xv-column__status" role="status">
      <span>{Math.min(completed + 1, total)} of {total}</span>
      <div
        className="xv-column__bar"
        role="progressbar"
        aria-label="Export progress"
        aria-valuemin={0}
        aria-valuenow={completed}
        aria-valuemax={total}
        style={{ '--xv-progress': total > 0 ? completed / total : 0 } as CSSProperties}
      />
      <button type="button" className="xv-column__status-btn" aria-label="Cancel export" onClick={cancelBatch}>
        <X size={10} />
      </button>
    </div>
  );
}

function DoneStatus({ batch }: { batch: ExportBatch }) {
  const { done } = batchSummary(batch);
  const reveal = getHost().exporter.reveal;
  const { target } = batch;
  // The foot's click handler dismisses the batch after this click.
  const onReveal = () => {
    if (!reveal || !target) return;
    const written = batch.jobs.find((job) => job.state === 'written');
    const opened = written ? reveal.open(target, written.name) : reveal.open(target);
    opened.catch((error: unknown) => console.error('Reveal failed', error));
  };
  return (
    <div className="xv-column__status" role="status">
      <span>{done} done</span>
      {reveal && target && <button type="button" className="xv-column__status-btn" onClick={onReveal}>{reveal.label}</button>}
    </div>
  );
}

function FailedStatus({ batch }: { batch: ExportBatch }) {
  const { done, failed } = batchSummary(batch);
  const [open, setOpen] = useState(false);
  const statusRef = useRef<HTMLDivElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const whyRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: PointerEvent) => {
      const inside = [statusRef, dialogRef].some((ref) => ref.current?.contains(event.target as Node));
      if (!inside) setOpen(false);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      event.preventDefault();
      setOpen(false);
      whyRef.current?.focus();
    };
    document.addEventListener('pointerdown', onPointerDown);
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('pointerdown', onPointerDown);
      document.removeEventListener('keydown', onKeyDown);
    };
  }, [open]);
  const rows = batch.jobs.filter((job) => job.state === 'failed' || job.state === 'skipped');
  return (
    <>
      <div ref={statusRef} className="xv-column__status" role="status">
        <span>{done > 0 ? `${done} done · ` : ''}{failed} failed</span>
        <button ref={whyRef} type="button" className="xv-column__status-btn" aria-expanded={open} onClick={() => setOpen((value) => !value)}>Why</button>
        <button type="button" className="xv-column__status-btn" aria-label="Dismiss export result" onClick={dismissBatch}>
          <X size={10} />
        </button>
      </div>
      {open && (
        <div ref={dialogRef} className="xv-column__failures xv-glass" role="dialog" aria-label="Export failures">
          {rows.map((job) => (
            <div key={job.id} className="xv-column__failure">
              <span className="name">{job.name}</span>
              <span className="reason">{job.state === 'skipped' ? 'skipped, exists' : job.error ?? 'failed'}</span>
              <button type="button" className="xv-column__status-btn" onClick={() => retryJob(job.id)}>Retry</button>
            </div>
          ))}
        </div>
      )}
    </>
  );
}
