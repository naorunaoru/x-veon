import { useEffect, useState } from 'react';
import { FloatingPanel } from '../FloatingPanel';
import { useAppStore } from '@/app/store';
import { useModelSizes } from '@/app/hooks/useModelSizes';
import { getHost } from '@/app/services/host';
import { demosaicMethodsFor, MODEL_SIZES } from '@/lib/catalog';
import type { DemosaicMethod, ModelSize } from '@/lib/types';
import { channelLabel } from '@/lib/channel';
import './Panels.css';
export function SettingsPanel() {
  const state = useAppStore();
  const file = state.files.find((f) => f.id === state.currentId);
  const { available, modelFor } = useModelSizes(file?.cfaType ?? null);
  const host = getHost();
  const [error, setError] = useState<string | null>(null);
  const [exportReason, setExportReason] = useState<string | null>(null);
  useEffect(() => {
    let active = true;
    setExportReason(null);
    void host.exporter.status().then(status => {
      if (active && !status.available) setExportReason(status.reason);
    }).catch(cause => {
      if (active) setExportReason(cause instanceof Error ? cause.message : String(cause));
    });
    return () => { active = false; };
  }, [host]);
  const methods = demosaicMethodsFor(file?.cfaType ?? null);
  const options = methods.map((method) => (
    <option key={method.id} value={method.id}>
      {method.label}
    </option>
  ));
  return (
    <FloatingPanel title="Settings" onClose={() => state.setOpenPanel(null)}>
      {file && (
        <fieldset
          disabled={file.editing === 'view-only'}
          style={{ border: 0, padding: 0, margin: 0 }}
        >
          <legend className="xv-field__label">Selected photo</legend>
          <div className="xv-field">
            <label htmlFor="photo-method">Photo demosaic method</label>
            <select
              id="photo-method"
              className="xv-select"
              value={file.edit.demosaicMethod ?? state.demosaicMethod}
              onChange={(e) => state.setFileDemosaicMethod(file.id, e.target.value as DemosaicMethod)}
            >
              {options}
            </select>
          </div>
          <div className="xv-field">
            <label htmlFor="photo-model">Photo model</label>
            <select
              id="photo-model"
              className="xv-select"
              value={file.edit.model?.size ?? state.modelSize}
              onChange={(e) => {
                try {
                  state.setFileModel(file.id, modelFor(e.target.value as ModelSize));
                  setError(null);
                } catch (cause) {
                  setError(cause instanceof Error ? cause.message : String(cause));
                }
              }}
            >
              {MODEL_SIZES.map((size) => (
                <option key={size} value={size} disabled={!available.has(size)}>
                  {size}
                </option>
              ))}
            </select>
          </div>
        </fieldset>
      )}
      {file?.editing !== 'saved' && file?.editingNote && (
        <p role="status">
          {file.editing === 'session' ? 'Session edits: ' : 'View only: '}
          {file.editingNote}
        </p>
      )}
      {file?.modelNote && <p role="status">{file.modelNote}</p>}
      <div className="xv-field">
        <label htmlFor="default-method">Default demosaic method</label>
        <select
          id="default-method"
          className="xv-select"
          value={state.demosaicMethod}
          onChange={(e) => state.setDemosaicMethod(e.target.value as DemosaicMethod)}
        >
          {demosaicMethodsFor(null).map((method) => (
            <option key={method.id} value={method.id}>
              {method.label}
            </option>
          ))}
        </select>
      </div>
      <div className="xv-field">
        <label htmlFor="default-model">Default model</label>
        <select
          id="default-model"
          className="xv-select"
          value={state.modelSize}
          onChange={(e) => state.setModelSize(e.target.value as ModelSize)}
        >
          {MODEL_SIZES.map((size) => (
            <option key={size} value={size} disabled={!available.has(size)}>
              {size}
            </option>
          ))}
        </select>
      </div>
      <div className="xv-field">
        <span className="xv-field__label">Output</span>
        <div className="xv-readout">
          HDR preview{' '}
          <b className={state.displayHdr ? 'hdr' : undefined}>{state.displayHdr ? 'ON' : 'OFF'}</b>
          {state.displayHdr && (
            <>
              {' '}
              · peak <b>{Math.round(state.displayHdrHeadroom * 100)} nits</b>
            </>
          )}
        </div>
      </div>
      {exportReason && (
        <div className="xv-field">
          <span className="xv-field__label">Export</span>
          <div className="xv-readout">{exportReason}</div>
        </div>
      )}
      <div className="xv-field">
        <span className="xv-field__label">Build</span>
        <div className="xv-readout" data-testid="xv-build">
          <b>{channelLabel(host.build.channel)}</b>{host.build.version && <> · {host.build.version}</>} · {host.build.sha}
          {host.build.date && <> · {host.build.date}</>}
          {host.channelLink && (
            <>
              <br />
              <a className="xv-readout__link" href={host.channelLink.href}>
                Open {host.channelLink.label}
              </a>{' '}
              · separate library
            </>
          )}
        </div>
      </div>
      {error && <p role="alert">{error}</p>}
    </FloatingPanel>
  );
}
