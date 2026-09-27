import * as SliderPrimitive from '@radix-ui/react-slider';
import { useAppStore } from '@/app/store';
import { sliderToZoom, zoomToSlider, formatZoom, minimumZoom, ZOOM_MAX } from '@/lib/zoom';
import './Slider.css'; // reuse the .xv-slider__track/__range/__thumb skin for the Radix slider
import './ZoomPill.css';

const PRESETS: { label: string; zoom: number }[] = [
  { label: '100%', zoom: 1 },
  { label: '200%', zoom: 2 },
];

export function ZoomPill() {
  const scale = useAppStore((s) => s.viewScale);
  const fitScale = useAppStore((s) => s.viewFitScale);
  const controls = useAppStore((s) => s.viewControls);

  const min = minimumZoom(fitScale);
  const max = Math.max(ZOOM_MAX, fitScale * 2);
  const readout = formatZoom(scale, fitScale);
  const isFit = readout === 'Fit';

  return (
    <div className="xv-zoompill xv-glass">
      <div className="xv-zoompill__presets">
        <button
          className={`xv-zoompill__preset${isFit ? ' is-active' : ''}`}
          onClick={() => controls?.resetView()}
        >
          Fit
        </button>
        {PRESETS.map((p) => (
          <button
            key={p.label}
            className={`xv-zoompill__preset${!isFit && Math.abs(scale - p.zoom) < p.zoom * 0.01 ? ' is-active' : ''}`}
            onClick={() => controls?.zoomTo(p.zoom)}
          >
            {p.label}
          </button>
        ))}
      </div>
      <span className="xv-zoompill__divider" />
      <SliderPrimitive.Root
        className="xv-zoompill__slider xv-slider__root"
        min={0} max={1} step={0.001}
        value={[zoomToSlider(scale, min, max)]}
        onValueChange={([t]) => controls?.zoomTo(sliderToZoom(t, min, max))}
        aria-label="Zoom"
      >
        <SliderPrimitive.Track className="xv-slider__track">
          <SliderPrimitive.Range className="xv-slider__range" />
        </SliderPrimitive.Track>
        <SliderPrimitive.Thumb className="xv-slider__thumb" onDoubleClick={() => controls?.zoomTo(1)} />
      </SliderPrimitive.Root>
      <span className="xv-zoompill__divider" />
      <span className="xv-zoompill__readout" data-testid="zoom-readout">{readout}</span>
    </div>
  );
}
