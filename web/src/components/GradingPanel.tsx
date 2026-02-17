import { useCallback, useMemo } from 'react';
import { RotateCcw } from 'lucide-react';
import { Slider } from '@/components/ui/slider';
import { Button } from '@/components/ui/button';
import { Histogram } from '@/components/Histogram';
import { cn } from '@/lib/utils';
import { useAppStore } from '@/store';
import { configFromPreset, type OpenDrtConfig } from '@/gl/opendrt-params';

interface ParamSliderProps {
  label: string;
  paramKey: keyof OpenDrtConfig;
  min: number;
  max: number;
  step: number;
  value: number;
  defaultValue: number;
  onChange: (key: keyof OpenDrtConfig, value: number) => void;
}

function ParamSlider({ label, paramKey, min, max, step, value, defaultValue, onChange }: ParamSliderProps) {
  const isModified = value !== defaultValue;

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <span className={`text-xs ${isModified ? 'text-foreground font-medium' : 'text-muted-foreground'}`}>
          {label}
        </span>
        <span className="text-xs tabular-nums text-muted-foreground w-12 text-right">
          {value.toFixed(step < 0.01 ? 3 : 2)}
        </span>
      </div>
      <Slider
        min={min}
        max={max}
        step={step}
        value={[value]}
        onValueChange={([v]) => onChange(paramKey, v)}
      />
    </div>
  );
}

// Slider definitions: [label, paramKey, min, max, step]
const TONE_PARAMS: [string, keyof OpenDrtConfig, number, number, number][] = [
  ['Contrast', 'tn_con', 0.5, 2.5, 0.01],
  ['Shoulder', 'tn_sh', 0.0, 1.0, 0.01],
  ['Toe', 'tn_toe', 0.0, 0.02, 0.001],
  ['Offset', 'tn_off', 0.0, 0.05, 0.001],
];

const COLOR_PARAMS: [string, keyof OpenDrtConfig, number, number, number][] = [
  ['Saturation', 'rs_sa', 0.0, 1.0, 0.01],
  ['Low contrast', 'tn_lcon', 0.0, 2.0, 0.01],
];

const BRILLIANCE_PARAMS: [string, keyof OpenDrtConfig, number, number, number][] = [
  ['Brilliance R', 'brl_r', -1.0, 1.0, 0.01],
  ['Brilliance G', 'brl_g', -1.0, 1.0, 0.01],
  ['Brilliance B', 'brl_b', -1.0, 1.0, 0.01],
];

export function GradingPanel() {
  const selectedFile = useAppStore((s) => s.files.find((f) => f.id === s.selectedFileId));
  const setFileLookPreset = useAppStore((s) => s.setFileLookPreset);
  const setFileOpenDrtOverride = useAppStore((s) => s.setFileOpenDrtOverride);
  const resetFileOpenDrtOverrides = useAppStore((s) => s.resetFileOpenDrtOverrides);

  const fileId = selectedFile?.id ?? null;
  const hasResult = selectedFile?.status === 'done';
  const lookPreset = selectedFile?.lookPreset ?? 'default';
  const overrides = selectedFile?.openDrtOverrides ?? {};

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

        {/* Color */}
        <section className="space-y-3">
          <h3 className="text-[11px] uppercase tracking-wider text-muted-foreground">Color</h3>
          {COLOR_PARAMS.map(([label, key, min, max, step]) => (
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
