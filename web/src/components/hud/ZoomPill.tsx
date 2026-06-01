import * as SliderPrimitive from '@radix-ui/react-slider';
import { useAppStore } from '@/store';
import { sliderToZoom, zoomToSlider, formatZoom } from '@/lib/zoom';
import './Slider.css'; // reuse the .xv-slider__track/__range/__thumb skin for the Radix slider
import './ZoomPill.css';

export function ZoomPill() {
  const scale = useAppStore((s) => s.viewScale);
  const fitScale = useAppStore((s) => s.viewFitScale);
  const controls = useAppStore((s) => s.viewControls);

  const readout = formatZoom(scale, fitScale);
  const isFit = readout === 'Fit';

  return (
    <div className="xv-zoompill xv-glass">
      <button
        className={`xv-zoompill__fit${isFit ? ' is-fit' : ''}`}
        onClick={() => controls?.resetView()}
      >
        Fit
      </button>
      <span className="xv-zoompill__divider" />
      <SliderPrimitive.Root
        className="xv-zoompill__slider xv-slider__root"
        min={0} max={1} step={0.001}
        value={[zoomToSlider(scale)]}
        onValueChange={([t]) => controls?.zoomTo(sliderToZoom(t))}
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
