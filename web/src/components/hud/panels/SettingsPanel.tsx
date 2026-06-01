import { FloatingPanel } from '../FloatingPanel';
import { useAppStore } from '@/store';
import { useProcessFile } from '@/hooks/useProcessFile';
import { getAvailableSizes, switchModelSize } from '@/pipeline/inference';
import type { CfaType, DemosaicMethod, ModelSize } from '@/pipeline/types';
import './Panels.css';

const DEMOSAIC_OPTIONS: { value: DemosaicMethod; label: string; cfa?: CfaType }[] = [
  { value: 'neural-net', label: 'X-veon' },
  { value: 'markesteijn3', label: 'Markesteijn (3-pass)', cfa: 'xtrans' },
  { value: 'markesteijn1', label: 'Markesteijn (1-pass)', cfa: 'xtrans' },
  { value: 'dht', label: 'DHT', cfa: 'xtrans' },
  { value: 'ahd', label: 'AHD', cfa: 'bayer' },
  { value: 'ppg', label: 'PPG', cfa: 'bayer' },
  { value: 'mhc', label: 'MHC', cfa: 'bayer' },
  { value: 'bilinear', label: 'Bilinear' },
];
const MODEL_SIZES: ModelSize[] = ['S', 'M', 'L'];

export function SettingsPanel() {
  const setOpenPanel = useAppStore((s) => s.setOpenPanel);
  const demosaicMethod = useAppStore((s) => s.demosaicMethod);
  const setDemosaicMethod = useAppStore((s) => s.setDemosaicMethod);
  const modelSize = useAppStore((s) => s.modelSize);
  const setModelSize = useAppStore((s) => s.setModelSize);
  const displayHdr = useAppStore((s) => s.displayHdr);
  const displayHdrHeadroom = useAppStore((s) => s.displayHdrHeadroom);
  const selectedFile = useAppStore((s) => s.files.find((f) => f.id === s.selectedFileId));
  const { processFile, isProcessing } = useProcessFile();

  const cfaType = selectedFile?.cfaType ?? null;
  const availableSizes = cfaType ? getAvailableSizes(cfaType) : new Set<ModelSize>(['S']);
  const availableMethods = DEMOSAIC_OPTIONS.filter((o) => !o.cfa || !cfaType || o.cfa === cfaType);

  const onModelClick = async (size: ModelSize) => {
    if (size === modelSize || !availableSizes.has(size)) return;
    setModelSize(size);
    await switchModelSize(size);
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
          {availableMethods.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
        </select>
      </div>

      <div className="xv-field">
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
      </div>

      <div className="xv-field">
        <span className="xv-field__label">Output</span>
        <div className="xv-readout">
          HDR preview <b className={displayHdr ? 'hdr' : undefined}>{displayHdr ? 'ON' : 'OFF'}</b>
          {displayHdr && <> · peak <b>{Math.round(displayHdrHeadroom * 100)} nits</b></>}<br />
          Tile <b>512</b> · Overlap <b>64</b>
        </div>
      </div>
    </FloatingPanel>
  );
}
