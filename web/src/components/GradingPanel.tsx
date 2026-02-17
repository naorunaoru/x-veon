import { useCallback, useMemo } from 'react';
import { RotateCcw } from 'lucide-react';
import { Slider } from '@/components/ui/slider';
import { Button } from '@/components/ui/button';
import { Histogram } from '@/components/Histogram';
import { cn } from '@/lib/utils';
import { useAppStore } from '@/store';
import { configFromPreset, type OpenDrtConfig } from '@/gl/opendrt-params';
import { estimateColorTemperature, findWbTempForCct, findWbTintForTint } from '@/pipeline/color-temperature';

interface ParamSliderProps {
  label: string;
  paramKey: keyof OpenDrtConfig;
  min: number;
  max: number;
  step: number;
  value: number;
  defaultValue: number;
  onChange: (key: keyof OpenDrtConfig, value: number) => void;
  /** Replaces the numeric value display with a human-readable label (e.g. "5500K"). */
  infoLabel?: string;
  /** Label shown at the left (min) end of the slider. */
  minLabel?: string;
  /** Label shown at the right (max) end of the slider. */
  maxLabel?: string;
  /** Custom inline style for the slider track (e.g. gradient background). */
  trackStyle?: React.CSSProperties;
}

function ParamSlider({ label, paramKey, min, max, step, value, defaultValue, onChange, infoLabel, minLabel, maxLabel, trackStyle }: ParamSliderProps) {
  const isModified = value !== defaultValue;

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <span className={`text-xs ${isModified ? 'text-foreground font-medium' : 'text-muted-foreground'}`}>
          {label}
        </span>
        <span className="text-xs tabular-nums text-muted-foreground text-right">
          {infoLabel ?? value.toFixed(step < 0.01 ? 3 : 2)}
        </span>
      </div>
      <Slider
        min={min}
        max={max}
        step={step}
        value={[value]}
        onValueChange={([v]) => onChange(paramKey, v)}
        trackStyle={trackStyle}
      />
      {(minLabel || maxLabel) && (
        <div className="flex justify-between -mt-0.5">
          <span className="text-[10px] text-muted-foreground">{minLabel}</span>
          <span className="text-[10px] text-muted-foreground">{maxLabel}</span>
        </div>
      )}
    </div>
  );
}

const TONE_PARAMS: [string, keyof OpenDrtConfig, number, number, number][] = [
  ['Contrast', 'tn_con', 0.5, 2.5, 0.01],
  ['Low contrast', 'tn_lcon', 0.0, 2.0, 0.01],
  ['Shoulder', 'tn_sh', 0.0, 1.0, 0.01],
  ['Toe', 'tn_toe', 0.0, 0.02, 0.001],
  ['Offset', 'tn_off', 0.0, 0.05, 0.001],
];

const BRILLIANCE_PARAMS: [string, keyof OpenDrtConfig, number, number, number][] = [
  ['Brilliance R', 'brl_r', -1.0, 1.0, 0.01],
  ['Brilliance G', 'brl_g', -1.0, 1.0, 0.01],
  ['Brilliance B', 'brl_b', -1.0, 1.0, 0.01],
];

function formatExposureBias(ev: number): string {
  if (ev === 0) return '0 EV';
  const sign = ev > 0 ? '+' : '';
  // Show common fractions for 1/3-stop increments
  const abs = Math.abs(ev);
  const thirds = Math.round(abs * 3);
  if (Math.abs(thirds / 3 - abs) < 0.01) {
    const whole = Math.floor(thirds / 3);
    const frac = thirds % 3;
    if (frac === 0) return `${sign}${ev > 0 ? whole : -whole} EV`;
    const fracStr = frac === 1 ? '1/3' : '2/3';
    if (whole === 0) return `${sign}${ev > 0 ? '' : '-'}${fracStr} EV`;
    return `${sign}${ev > 0 ? whole : -whole} ${fracStr} EV`;
  }
  return `${sign}${ev.toFixed(1)} EV`;
}

export function GradingPanel() {
  const selectedFile = useAppStore((s) => s.files.find((f) => f.id === s.selectedFileId));
  const setFileLookPreset = useAppStore((s) => s.setFileLookPreset);
  const setFileOpenDrtOverride = useAppStore((s) => s.setFileOpenDrtOverride);
  const resetFileOpenDrtOverrides = useAppStore((s) => s.resetFileOpenDrtOverrides);

  const fileId = selectedFile?.id ?? null;
  const hasResult = selectedFile?.status === 'done';
  const lookPreset = selectedFile?.lookPreset ?? 'default';
  const overrides = selectedFile?.openDrtOverrides ?? {};
  const resultMeta = selectedFile?.result?.metadata;
  const exportData = selectedFile?.result?.exportData;

  const baseConfig = useMemo(() => configFromPreset(lookPreset), [lookPreset]);

  const effectiveValue = useCallback(
    (key: keyof OpenDrtConfig): number => {
      return (overrides[key] as number) ?? (baseConfig[key] as number);
    },
    [overrides, baseConfig],
  );

  const handleChange = useCallback(
    (key: keyof OpenDrtConfig, value: number) => {
      if (fileId) setFileOpenDrtOverride(fileId, key, value);
    },
    [fileId, setFileOpenDrtOverride],
  );

  const handleExposureChange = useCallback(
    (_: keyof OpenDrtConfig, absEv: number) => {
      if (!fileId || !resultMeta) return;
      setFileOpenDrtOverride(fileId, 'exposure', absEv - resultMeta.exposureBias);
    },
    [fileId, resultMeta, setFileOpenDrtOverride],
  );

  const handleTempChange = useCallback(
    (_: keyof OpenDrtConfig, cct: number) => {
      if (!fileId || !exportData?.camToXyz) return;
      const wbTemp = findWbTempForCct(cct, exportData.wbCoeffs, exportData.camToXyz);
      setFileOpenDrtOverride(fileId, 'wb_temp', wbTemp);
    },
    [fileId, exportData, setFileOpenDrtOverride],
  );

  const handleTintChange = useCallback(
    (_: keyof OpenDrtConfig, tintVal: number) => {
      if (!fileId || !exportData?.camToXyz) return;
      const wbTint = findWbTintForTint(tintVal, exportData.wbCoeffs, exportData.camToXyz);
      setFileOpenDrtOverride(fileId, 'wb_tint', wbTint);
    },
    [fileId, exportData, setFileOpenDrtOverride],
  );

  // Compute effective exposure / CCT / tint from slider adjustments + camera WB
  const shootingInfo = useMemo(() => {
    if (!resultMeta || !exportData?.camToXyz) return null;
    const wb = exportData.wbCoeffs;
    const camToXyz = exportData.camToXyz;
    const temp = effectiveValue('wb_temp');
    const tint = effectiveValue('wb_tint');
    const exposure = effectiveValue('exposure');

    // Base values (no adjustment) — used as slider default positions
    const baseWb = new Float32Array([wb[0], 1.0, wb[2]]);
    const { temp: baseTempK, tint: baseTint } = estimateColorTemperature(baseWb, camToXyz);

    // Decouple CCT and tint estimation: compute each from its own slider only.
    // R/B gain doesn't trace the Planckian locus exactly, so applying both
    // together causes cross-talk (temp changes tint display and vice versa).
    const tempWb = new Float32Array([
      wb[0] * Math.pow(2, temp), 1.0, wb[2] * Math.pow(2, -temp),
    ]);
    const tintWb = new Float32Array([
      wb[0], Math.pow(2, -tint), wb[2],
    ]);
    const { temp: cct } = estimateColorTemperature(tempWb, camToXyz);
    const { tint: cctTint } = estimateColorTemperature(tintWb, camToXyz);

    return {
      baseBias: resultMeta.exposureBias,
      exposureEv: resultMeta.exposureBias + exposure,
      exposure: formatExposureBias(resultMeta.exposureBias + exposure),
      baseTempK,
      tempK: cct,
      temp: `${cct}K`,
      baseTint,
      tintValue: cctTint,
      tint: `${cctTint > 0 ? '+' : ''}${cctTint}`,
    };
  }, [resultMeta, exportData, effectiveValue]);

  const hasOverrides = Object.keys(overrides).length > 0;

  return (
    <aside className="flex flex-col h-screen w-[240px] min-w-[240px] border-l border-border bg-background">
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <span className="text-sm font-medium">Grading</span>
        {fileId && (
          <Button
            variant="ghost"
            size="sm"
            className="h-6 w-6 p-0"
            disabled={!hasOverrides}
            onClick={() => resetFileOpenDrtOverrides(fileId)}
            title="Reset to preset"
          >
            <RotateCcw className="h-3.5 w-3.5" />
          </Button>
        )}
      </div>

      <div className={cn("flex-1 overflow-y-auto px-4 py-3 space-y-5", !hasResult && "opacity-50 pointer-events-none")}>
        <Histogram />

        {/* Exposure */}
        <section className="space-y-3">
          <h3 className="text-[11px] uppercase tracking-wider text-muted-foreground">Exposure</h3>
          <ParamSlider
            label="Exposure"
            paramKey="exposure"
            min={-5}
            max={5}
            step={0.01}
            value={shootingInfo?.exposureEv ?? effectiveValue('exposure')}
            defaultValue={shootingInfo?.baseBias ?? 0}
            onChange={handleExposureChange}
            infoLabel={shootingInfo?.exposure}
            minLabel="-5 EV"
            maxLabel="+5 EV"
          />
        </section>

        {/* White Balance */}
        <section className="space-y-3">
          <h3 className="text-[11px] uppercase tracking-wider text-muted-foreground">White Balance</h3>
          <ParamSlider
            label="Temperature"
            paramKey="wb_temp"
            min={2000}
            max={12000}
            step={50}
            value={shootingInfo?.tempK ?? 5500}
            defaultValue={shootingInfo?.baseTempK ?? 5500}
            onChange={handleTempChange}
            infoLabel={shootingInfo?.temp}
            minLabel="2000K"
            maxLabel="12000K"
            trackStyle={{ background: 'linear-gradient(to right, #5B8FC9, #E8A438)' }}
          />
          <ParamSlider
            label="Tint"
            paramKey="wb_tint"
            min={-150}
            max={150}
            step={1}
            value={shootingInfo?.tintValue ?? 0}
            defaultValue={shootingInfo?.baseTint ?? 0}
            onChange={handleTintChange}
            infoLabel={shootingInfo?.tint}
            minLabel="Magenta"
            maxLabel="Green"
            trackStyle={{ background: 'linear-gradient(to right, #C850C0, #4BA84D)' }}
          />
        </section>

        {/* Detail */}
        <section className="space-y-3">
          <h3 className="text-[11px] uppercase tracking-wider text-muted-foreground">Detail</h3>
          <ParamSlider
            label="Sharpening"
            paramKey="sharpen_amount"
            min={0}
            max={2}
            step={0.01}
            value={effectiveValue('sharpen_amount')}
            defaultValue={baseConfig.sharpen_amount}
            onChange={handleChange}
          />
        </section>

        {/* Look Preset */}
        <section className="space-y-1.5">
          <h3 className="text-[11px] uppercase tracking-wider text-muted-foreground">Look</h3>
          <div className="flex rounded-md bg-muted p-0.5">
            {([['default', 'Default'], ['base', 'Base'], ['flat', 'Flat']] as const).map(([value, label]) => (
              <button
                key={value}
                className={cn(
                  'flex-1 rounded-sm px-2 py-1 text-xs font-medium transition-colors',
                  lookPreset === value
                    ? 'bg-background text-foreground shadow-sm'
                    : 'text-muted-foreground hover:text-foreground',
                )}
                onClick={() => fileId && setFileLookPreset(fileId, value)}
              >
                {label}
              </button>
            ))}
          </div>
        </section>

        {/* Tone */}
        <section className="space-y-3">
          <h3 className="text-[11px] uppercase tracking-wider text-muted-foreground">Tone</h3>
          {TONE_PARAMS.map(([label, key, min, max, step]) => (
            <ParamSlider
              key={key}
              label={label}
              paramKey={key}
              min={min}
              max={max}
              step={step}
              value={effectiveValue(key)}
              defaultValue={baseConfig[key] as number}
              onChange={handleChange}
            />
          ))}
        </section>

        {/* Brilliance */}
        <section className="space-y-3">
          <h3 className="text-[11px] uppercase tracking-wider text-muted-foreground">Brilliance</h3>
          {BRILLIANCE_PARAMS.map(([label, key, min, max, step]) => (
            <ParamSlider
              key={key}
              label={label}
              paramKey={key}
              min={min}
              max={max}
              step={step}
              value={effectiveValue(key)}
              defaultValue={baseConfig[key] as number}
              onChange={handleChange}
            />
          ))}
        </section>
      </div>
    </aside>
  );
}
