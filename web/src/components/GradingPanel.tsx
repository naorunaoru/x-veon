import { useCallback, useMemo, useState } from 'react';
import { ChevronRight, RotateCcw } from 'lucide-react';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Button } from '@/components/ui/button';
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from '@/components/ui/select';
import { Histogram } from '@/components/Histogram';
import { cn } from '@/lib/utils';
import { useAppStore } from '@/store';
import { configFromPreset, DEFAULT_PREPROCESS, TONESCALE_PRESETS, type OpenDrtConfig, type PreProcessConfig, type TonescalePreset } from '@/gl/opendrt-params';
import { estimateColorTemperature, findWbTempForCct, findWbTintForTint } from '@/pipeline/color-temperature';

interface ParamSliderProps {
  label: string;
  paramKey: string;
  min: number;
  max: number;
  step: number;
  value: number;
  defaultValue: number;
  onChange: (key: string, value: number) => void;
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

type ParamDef = [string, keyof OpenDrtConfig, number, number, number];

const TONE_PARAMS: ParamDef[] = [
  ['Contrast', 'tn_con', 0.5, 2.5, 0.01],
  ['Shoulder', 'tn_sh', 0.0, 1.0, 0.01],
  ['Toe', 'tn_toe', 0.0, 0.02, 0.001],
  ['Offset', 'tn_off', 0.0, 0.05, 0.001],
];

const TONE_LCON_PARAMS: ParamDef[] = [
  ['Amount', 'tn_lcon', 0.0, 2.0, 0.01],
  ['Width', 'tn_lcon_w', 0.0, 2.0, 0.01],
  ['Per-Channel', 'tn_lcon_pc', 0.0, 1.0, 0.01],
];

const TONE_HCON_PARAMS: ParamDef[] = [
  ['Amount', 'tn_hcon', -1.0, 1.0, 0.01],
  ['Pivot', 'tn_hcon_pv', 0.0, 4.0, 0.01],
  ['Strength', 'tn_hcon_st', 0.0, 8.0, 0.01],
];

const PURITY_COMPRESS_PARAMS: ParamDef[] = [
  ['Compress R', 'pt_r', 0.0, 5.0, 0.01],
  ['Compress G', 'pt_g', 0.0, 5.0, 0.01],
  ['Compress B', 'pt_b', 0.0, 5.0, 0.01],
  ['Range Low', 'pt_rng_low', 0.0, 1.0, 0.01],
  ['Range High', 'pt_rng_high', 0.0, 1.0, 0.01],
];

const PURITY_RENDER_PARAMS: ParamDef[] = [
  ['Render Strength', 'rs_sa', 0.0, 1.0, 0.01],
  ['Red Weight', 'rs_rw', 0.0, 1.0, 0.01],
  ['Blue Weight', 'rs_bw', 0.0, 1.0, 0.01],
];

const MID_PURITY_PARAMS: ParamDef[] = [
  ['Low', 'ptm_low', -1.0, 1.0, 0.01],
  ['Low Strength', 'ptm_low_st', 0.0, 1.0, 0.01],
  ['High', 'ptm_high', -1.0, 1.0, 0.01],
  ['High Strength', 'ptm_high_st', 0.0, 1.0, 0.01],
];

const BRILLIANCE_PARAMS: ParamDef[] = [
  ['Red', 'brl_r', -1.0, 1.0, 0.01],
  ['Green', 'brl_g', -1.0, 1.0, 0.01],
  ['Blue', 'brl_b', -1.0, 1.0, 0.01],
  ['Cyan', 'brl_c', -1.0, 1.0, 0.01],
  ['Magenta', 'brl_m', -1.0, 1.0, 0.01],
  ['Yellow', 'brl_y', -1.0, 1.0, 0.01],
  ['Range', 'brl_rng', 0.0, 1.0, 0.01],
];

const HUE_SHIFT_RGB_PARAMS: ParamDef[] = [
  ['Red', 'hs_r', -1.0, 2.0, 0.01],
  ['Green', 'hs_g', -1.0, 2.0, 0.01],
  ['Blue', 'hs_b', -1.0, 2.0, 0.01],
  ['Range', 'hs_rgb_rng', 0.0, 4.0, 0.01],
];

const HUE_SHIFT_CMY_PARAMS: ParamDef[] = [
  ['Cyan', 'hs_c', -1.0, 2.0, 0.01],
  ['Magenta', 'hs_m', -1.0, 2.0, 0.01],
  ['Yellow', 'hs_y', -1.0, 2.0, 0.01],
];

/** Collapsible accordion section with optional enable toggle in the heading. */
function AccordionSection({
  title,
  defaultOpen = false,
  toggleKey,
  toggleChecked,
  onToggle,
  extra,
  children,
}: {
  title: string;
  defaultOpen?: boolean;
  toggleKey?: keyof OpenDrtConfig;
  toggleChecked?: boolean;
  onToggle?: (key: keyof OpenDrtConfig, value: boolean) => void;
  extra?: React.ReactNode;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section>
      <button
        type="button"
        className="flex items-center gap-1.5 w-full text-left group"
        onClick={() => setOpen((o) => !o)}
      >
        <ChevronRight className={cn('h-3 w-3 text-muted-foreground transition-transform', open && 'rotate-90')} />
        <h3 className="text-[11px] uppercase tracking-wider text-muted-foreground flex-1">{title}</h3>
        {extra}
        {toggleKey != null && (
          <Switch
            checked={toggleChecked}
            onCheckedChange={(v) => { onToggle?.(toggleKey, v); }}
            onClick={(e) => e.stopPropagation()}
            className="ml-auto"
          />
        )}
      </button>
      {open && <div className="space-y-3 mt-3">{children}</div>}
    </section>
  );
}

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
  const setFilePreProcessOverride = useAppStore((s) => s.setFilePreProcessOverride);
  const resetFilePreProcessOverrides = useAppStore((s) => s.resetFilePreProcessOverrides);
  const displayHdr = useAppStore((s) => s.displayHdr);
  const displayHdrHeadroom = useAppStore((s) => s.displayHdrHeadroom);

  const fileId = selectedFile?.id ?? null;
  const hasResult = selectedFile?.status === 'done';
  const lookPreset = selectedFile?.lookPreset ?? 'default';
  const overrides = selectedFile?.openDrtOverrides ?? {};
  const preProcessOverrides = selectedFile?.preProcessOverrides ?? {};
  const resultMeta = selectedFile?.result?.metadata;
  const exportData = selectedFile?.result?.exportData;

  const hdrHeadroom = displayHdr ? displayHdrHeadroom : undefined;
  const baseConfig = useMemo(() => configFromPreset(lookPreset, hdrHeadroom), [lookPreset, hdrHeadroom]);

  const effectiveValue = useCallback(
    <K extends keyof OpenDrtConfig>(key: K): OpenDrtConfig[K] => {
      return (overrides[key] ?? baseConfig[key]) as OpenDrtConfig[K];
    },
    [overrides, baseConfig],
  );

  const effectivePreProcess = useCallback(
    (key: keyof PreProcessConfig): number => {
      return preProcessOverrides[key] ?? DEFAULT_PREPROCESS[key];
    },
    [preProcessOverrides],
  );

  const handleChange = useCallback(
    (key: string, value: number) => {
      if (fileId) setFileOpenDrtOverride(fileId, key as keyof OpenDrtConfig, value);
    },
    [fileId, setFileOpenDrtOverride],
  );

  const handlePreProcessChange = useCallback(
    (key: string, value: number) => {
      if (fileId) setFilePreProcessOverride(fileId, key as keyof PreProcessConfig, value);
    },
    [fileId, setFilePreProcessOverride],
  );

  const handleExposureChange = useCallback(
    (_: string, absEv: number) => {
      if (!fileId || !resultMeta) return;
      setFilePreProcessOverride(fileId, 'exposure', absEv - resultMeta.exposureBias);
    },
    [fileId, resultMeta, setFilePreProcessOverride],
  );

  const handleTempChange = useCallback(
    (_: string, cct: number) => {
      if (!fileId || !exportData?.camToXyz) return;
      const wbTemp = findWbTempForCct(cct, exportData.wbCoeffs, exportData.camToXyz);
      setFilePreProcessOverride(fileId, 'wb_temp', wbTemp);
    },
    [fileId, exportData, setFilePreProcessOverride],
  );

  const handleTintChange = useCallback(
    (_: string, tintVal: number) => {
      if (!fileId || !exportData?.camToXyz) return;
      const wbTint = findWbTintForTint(tintVal, exportData.wbCoeffs, exportData.camToXyz);
      setFilePreProcessOverride(fileId, 'wb_tint', wbTint);
    },
    [fileId, exportData, setFilePreProcessOverride],
  );

  // Compute effective exposure / CCT / tint from slider adjustments + camera WB
  const shootingInfo = useMemo(() => {
    if (!resultMeta || !exportData?.camToXyz) return null;
    const wb = exportData.wbCoeffs;
    const camToXyz = exportData.camToXyz;
    const temp = effectivePreProcess('wb_temp');
    const tint = effectivePreProcess('wb_tint');
    const exposure = effectivePreProcess('exposure');

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
  }, [resultMeta, exportData, effectivePreProcess]);

  const handleToggle = useCallback(
    (key: keyof OpenDrtConfig, value: boolean) => {
      if (fileId) setFileOpenDrtOverride(fileId, key, value);
    },
    [fileId, setFileOpenDrtOverride],
  );

  const hasOverrides = Object.keys(overrides).length > 0 || Object.keys(preProcessOverrides).length > 0;

  const renderParams = (params: ParamDef[]) =>
    params.map(([label, key, min, max, step]) => (
      <ParamSlider
        key={key}
        label={label}
        paramKey={key}
        min={min}
        max={max}
        step={step}
        value={effectiveValue(key) as number}
        defaultValue={baseConfig[key] as number}
        onChange={handleChange}
      />
    ));

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
            onClick={() => { resetFileOpenDrtOverrides(fileId); resetFilePreProcessOverrides(fileId); }}
            title="Reset to preset"
          >
            <RotateCcw className="h-3.5 w-3.5" />
          </Button>
        )}
      </div>

      <div className={cn("flex-1 overflow-y-auto px-4 py-3 space-y-5", !hasResult && "opacity-50 pointer-events-none")}>
        <Histogram />

        {/* Display Peak Luminance */}
        <ParamSlider
          label="Peak Luminance"
          paramKey="peak_luminance"
          min={100}
          max={2000}
          step={10}
          value={effectiveValue('peak_luminance') as number}
          defaultValue={displayHdr ? Math.round(displayHdrHeadroom * 100) : 100}
          onChange={handleChange}
          infoLabel={`${Math.round(effectiveValue('peak_luminance') as number)} nits`}
        />

        {/* Exposure */}
        <section className="space-y-3">
          <h3 className="text-[11px] uppercase tracking-wider text-muted-foreground">Exposure</h3>
          <ParamSlider
            label="Exposure"
            paramKey="exposure"
            min={-5}
            max={5}
            step={0.01}
            value={shootingInfo?.exposureEv ?? effectivePreProcess('exposure')}
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
            value={effectivePreProcess('sharpen_amount')}
            defaultValue={DEFAULT_PREPROCESS.sharpen_amount}
            onChange={handlePreProcessChange}
          />
        </section>

        {/* Look Preset */}
        <div className="flex items-center gap-3">
          <span className="text-xs text-muted-foreground w-14 flex-shrink-0">Look</span>
          <div className="flex-1">
            <Select value={lookPreset} onValueChange={(v) => fileId && setFileLookPreset(fileId, v as typeof lookPreset)}>
              <SelectTrigger className="h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {([['default', 'Default'], ['colorful', 'Colorful'], ['umbra', 'Umbra'], ['base', 'Base'], ['flat', 'Flat']] as const).map(([value, label]) => (
                  <SelectItem key={value} value={value}>{label}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        {/* Tonescale Preset */}
        <div className="flex items-center gap-3">
          <span className="text-xs text-muted-foreground w-14 flex-shrink-0">Curve</span>
          <div className="flex-1">
            <Select
              value=""
              onValueChange={(v) => {
                if (!fileId || !v) return;
                const preset = TONESCALE_PRESETS[v as TonescalePreset];
                if (!preset) return;
                const entries = Object.entries(preset.overrides) as [keyof OpenDrtConfig, OpenDrtConfig[keyof OpenDrtConfig]][];
                for (const [k, val] of entries) {
                  setFileOpenDrtOverride(fileId, k, val);
                }
              }}
            >
              <SelectTrigger className="h-8 text-xs">
                <SelectValue placeholder="Presets…" />
              </SelectTrigger>
              <SelectContent>
                {Object.entries(TONESCALE_PRESETS).map(([key, { label }]) => (
                  <SelectItem key={key} value={key}>{label}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        {/* ── Tonescale ── */}
        <AccordionSection title="Tonescale">
          <ParamSlider
            label="Middle Grey"
            paramKey="tn_lg"
            min={3.0}
            max={25.0}
            step={0.1}
            value={effectiveValue('tn_lg')}
            defaultValue={baseConfig.tn_lg as number}
            onChange={handleChange}
            infoLabel={`${effectiveValue('tn_lg').toFixed(1)} nits`}
          />
          {renderParams(TONE_PARAMS)}

          {/* Low Contrast sub-group */}
          <div className="space-y-2 pt-1">
            <div className="flex items-center justify-between">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Low Contrast</span>
              <Switch
                checked={effectiveValue('tn_lcon_enable')}
                onCheckedChange={(v) => handleToggle('tn_lcon_enable', v)}
              />
            </div>
            {renderParams(TONE_LCON_PARAMS)}
          </div>

          {/* High Contrast sub-group */}
          <div className="space-y-2 pt-1">
            <div className="flex items-center justify-between">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground">High Contrast</span>
              <Switch
                checked={effectiveValue('tn_hcon_enable')}
                onCheckedChange={(v) => handleToggle('tn_hcon_enable', v)}
              />
            </div>
            {renderParams(TONE_HCON_PARAMS)}
          </div>

          {/* Creative White */}
          <div className="space-y-2 pt-1">
            <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Creative White</span>
            <ParamSlider
              label="Warmth"
              paramKey="cwp"
              min={0.0}
              max={1.0}
              step={0.01}
              value={effectiveValue('cwp')}
              defaultValue={baseConfig.cwp as number}
              onChange={handleChange}
              minLabel="D65"
              maxLabel="D50"
            />
            <ParamSlider
              label="Range"
              paramKey="cwp_rng"
              min={0.0}
              max={1.0}
              step={0.01}
              value={effectiveValue('cwp_rng')}
              defaultValue={baseConfig.cwp_rng as number}
              onChange={handleChange}
            />
          </div>
        </AccordionSection>

        {/* ── Purity ── */}
        <AccordionSection title="Purity">
          {renderParams(PURITY_RENDER_PARAMS)}
          {renderParams(PURITY_COMPRESS_PARAMS)}

          {/* Purity Compress Low */}
          <div className="space-y-2 pt-1">
            <div className="flex items-center justify-between">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Compress Low</span>
              <Switch
                checked={effectiveValue('ptl_enable')}
                onCheckedChange={(v) => handleToggle('ptl_enable', v)}
              />
            </div>
          </div>

          {/* Mid Purity */}
          <div className="space-y-2 pt-1">
            <div className="flex items-center justify-between">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Mid Purity</span>
              <Switch
                checked={effectiveValue('ptm_enable')}
                onCheckedChange={(v) => handleToggle('ptm_enable', v)}
              />
            </div>
            {renderParams(MID_PURITY_PARAMS)}
          </div>
        </AccordionSection>

        {/* ── Brilliance ── */}
        <AccordionSection
          title="Brilliance"
          toggleKey="brl_enable"
          toggleChecked={effectiveValue('brl_enable')}
          onToggle={handleToggle}
        >
          {renderParams(BRILLIANCE_PARAMS)}
        </AccordionSection>

        {/* ── Hue Shift ── */}
        <AccordionSection title="Hue Shift">
          {/* RGB */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground">RGB</span>
              <Switch
                checked={effectiveValue('hs_rgb_enable')}
                onCheckedChange={(v) => handleToggle('hs_rgb_enable', v)}
              />
            </div>
            {renderParams(HUE_SHIFT_RGB_PARAMS)}
          </div>

          {/* CMY */}
          <div className="space-y-2 pt-1">
            <div className="flex items-center justify-between">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground">CMY</span>
              <Switch
                checked={effectiveValue('hs_cmy_enable')}
                onCheckedChange={(v) => handleToggle('hs_cmy_enable', v)}
              />
            </div>
            {renderParams(HUE_SHIFT_CMY_PARAMS)}
          </div>

          {/* Hue Contrast */}
          <div className="space-y-2 pt-1">
            <div className="flex items-center justify-between">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Hue Contrast</span>
              <Switch
                checked={effectiveValue('hc_enable')}
                onCheckedChange={(v) => handleToggle('hc_enable', v)}
              />
            </div>
            <ParamSlider
              label="Red"
              paramKey="hc_r"
              min={0.0}
              max={2.0}
              step={0.01}
              value={effectiveValue('hc_r')}
              defaultValue={baseConfig.hc_r as number}
              onChange={handleChange}
            />
          </div>
        </AccordionSection>
      </div>
    </aside>
  );
}
