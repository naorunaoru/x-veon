import { FloatingPanel } from '../FloatingPanel';
import { useAppStore } from '@/app/store';
import { useProcessing } from '@/app/hooks/useProcessing';
import { useModelSizes } from '@/app/hooks/useModelSizes';
import { demosaicMethodsFor, MODEL_SIZES } from '@/lib/catalog';
import type { DemosaicMethod, ModelSize } from '@/lib/types';
import { BUILD, channelLabel, otherChannelLink } from '@/lib/channel';
import './Panels.css';

export function SettingsPanel() {
  const setOpenPanel = useAppStore((s) => s.setOpenPanel);
  const demosaicMethod = useAppStore((s) => s.demosaicMethod);
  const setDemosaicMethod = useAppStore((s) => s.setDemosaicMethod);
  const modelSize = useAppStore((s) => s.modelSize);
  const setModelSize = useAppStore((s) => s.setModelSize);
  const displayHdr = useAppStore((s) => s.displayHdr);
  const displayHdrHeadroom = useAppStore((s) => s.displayHdrHeadroom);
  const selectedFile = useAppStore((s) => s.files.find((f) => f.id === s.selectedFileId));
  const { processFile, isProcessing } = useProcessing();
  const other = otherChannelLink(BUILD.channel);

  const cfaType = selectedFile?.cfaType ?? null;
  const { available: availableSizes, switchTo } = useModelSizes(cfaType);
  const availableMethods = demosaicMethodsFor(cfaType);

  const onModelClick = async (size: ModelSize) => {
    if (size === modelSize || !availableSizes.has(size)) return;
    const previous = modelSize;
    setModelSize(size);
    try {
      await switchTo(size);
    } catch (e) {
      console.error('Model switch failed:', e);
      setModelSize(previous);
      return;
    }
    const file = useAppStore.getState().files.find((f) => f.id === useAppStore.getState().selectedFileId);
    if (file && (file.status === 'done' || file.status === 'error') && !isProcessing && demosaicMethod === 'neural-net') {
      processFile(file.id);
    }
  };

  return (
    <FloatingPanel title="Settings" onClose={() => setOpenPanel(null)}>
      <div className="xv-field">
        <label className="xv-field__label" htmlFor="xv-demosaic">Demosaic method</label>
        <select
          id="xv-demosaic" className="xv-select"
          aria-label="Demosaic method"
          value={demosaicMethod}
          onChange={(e) => setDemosaicMethod(e.target.value as DemosaicMethod)}
        >
          {availableMethods.map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
        </select>
      </div>

      {availableSizes.size > 1 && <div className="xv-field">
        <span className="xv-field__label">Model</span>
        <div className="xv-seg">
          {MODEL_SIZES.map((s) => (
            <button
              key={s}
              className={`xv-seg__btn${modelSize === s ? ' is-active' : ''}`}
              disabled={!availableSizes.has(s)}
              onClick={() => onModelClick(s)}
            >
              {s}
            </button>
          ))}
        </div>
      </div>}

      <div className="xv-field">
        <span className="xv-field__label">Output</span>
        <div className="xv-readout">
          HDR preview <b className={displayHdr ? 'hdr' : undefined}>{displayHdr ? 'ON' : 'OFF'}</b>
          {displayHdr && <> · peak <b>{Math.round(displayHdrHeadroom * 100)} nits</b></>}
        </div>
      </div>

      <div className="xv-field">
        <span className="xv-field__label">Build</span>
        <div className="xv-readout" data-testid="xv-build">
          <b>{channelLabel(BUILD.channel)}</b> · {BUILD.sha}{BUILD.date && <> · {BUILD.date}</>}
          {other && (
            <>
              <br />
              <a className="xv-readout__link" href={other.href}>Open {other.label}</a> · separate library
            </>
          )}
        </div>
      </div>
    </FloatingPanel>
  );
}
