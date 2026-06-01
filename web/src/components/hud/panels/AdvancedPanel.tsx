import { FloatingPanel } from '../FloatingPanel';
import { Slider } from '../Slider';
import { Collapsible } from '../Collapsible';
import { Toggle } from '../Toggle';
import { useGrading } from '@/hooks/useGrading';
import { useAppStore } from '@/store';
import { isSectionModified, SECTION_KEYS } from '@/lib/grading/sections';
import type { OpenDrtConfig } from '@/gl/opendrt-params';
import './Panels.css';

type Grading = ReturnType<typeof useGrading>;
type DrtKey = keyof OpenDrtConfig;
interface Param { label: string; key: DrtKey; min: number; max: number; step: number; }

interface Group {
  title: string;
  blocks: { subTitle?: string; subEnableKey?: DrtKey; params: Param[] }[];
}

const GROUPS: Group[] = [
  {
    title: 'Tonescale',
    blocks: [
      { params: [
        { label: 'Middle Grey', key: 'tn_lg', min: 3, max: 25, step: 0.1 },
        { label: 'Offset', key: 'tn_off', min: 0, max: 0.05, step: 0.001 },
      ] },
      { subTitle: 'Low Contrast', subEnableKey: 'tn_lcon_enable', params: [
        { label: 'Width', key: 'tn_lcon_w', min: 0, max: 2, step: 0.01 },
        { label: 'Per-Channel', key: 'tn_lcon_pc', min: 0, max: 1, step: 0.01 },
      ] },
      { subTitle: 'High Contrast', subEnableKey: 'tn_hcon_enable', params: [
        { label: 'Amount', key: 'tn_hcon', min: -1, max: 1, step: 0.01 },
        { label: 'Pivot', key: 'tn_hcon_pv', min: 0, max: 4, step: 0.01 },
        { label: 'Strength', key: 'tn_hcon_st', min: 0, max: 8, step: 0.01 },
      ] },
      { subTitle: 'Creative White', params: [
        { label: 'Warmth', key: 'cwp', min: 0, max: 1, step: 0.01 },
        { label: 'Range', key: 'cwp_rng', min: 0, max: 1, step: 0.01 },
      ] },
    ],
  },
  {
    title: 'Purity',
    blocks: [
      { subTitle: 'Render', params: [
        { label: 'Render Strength', key: 'rs_sa', min: 0, max: 1, step: 0.01 },
        { label: 'Red Weight', key: 'rs_rw', min: 0, max: 1, step: 0.01 },
        { label: 'Blue Weight', key: 'rs_bw', min: 0, max: 1, step: 0.01 },
      ] },
      { subTitle: 'Compress', params: [
        { label: 'Compress R', key: 'pt_r', min: 0, max: 5, step: 0.01 },
        { label: 'Compress G', key: 'pt_g', min: 0, max: 5, step: 0.01 },
        { label: 'Compress B', key: 'pt_b', min: 0, max: 5, step: 0.01 },
        { label: 'Range Low', key: 'pt_rng_low', min: 0, max: 1, step: 0.01 },
        { label: 'Range High', key: 'pt_rng_high', min: 0, max: 1, step: 0.01 },
      ] },
      { subTitle: 'Compress Low', subEnableKey: 'ptl_enable', params: [] },
      { subTitle: 'Mid Purity', subEnableKey: 'ptm_enable', params: [
        { label: 'Low', key: 'ptm_low', min: -1, max: 1, step: 0.01 },
        { label: 'Low Strength', key: 'ptm_low_st', min: 0, max: 1, step: 0.01 },
        { label: 'High', key: 'ptm_high', min: -1, max: 1, step: 0.01 },
        { label: 'High Strength', key: 'ptm_high_st', min: 0, max: 1, step: 0.01 },
      ] },
    ],
  },
  {
    title: 'Brilliance C/M/Y',
    blocks: [
      { params: [
        { label: 'Cyan', key: 'brl_c', min: -1, max: 1, step: 0.01 },
        { label: 'Magenta', key: 'brl_m', min: -1, max: 1, step: 0.01 },
        { label: 'Yellow', key: 'brl_y', min: -1, max: 1, step: 0.01 },
        { label: 'Range', key: 'brl_rng', min: 0, max: 1, step: 0.01 },
      ] },
    ],
  },
  {
    title: 'Hue Shift',
    blocks: [
      { subTitle: 'RGB', subEnableKey: 'hs_rgb_enable', params: [
        { label: 'Red', key: 'hs_r', min: -1, max: 2, step: 0.01 },
        { label: 'Green', key: 'hs_g', min: -1, max: 2, step: 0.01 },
        { label: 'Blue', key: 'hs_b', min: -1, max: 2, step: 0.01 },
        { label: 'Range', key: 'hs_rgb_rng', min: 0, max: 4, step: 0.01 },
      ] },
      { subTitle: 'CMY', subEnableKey: 'hs_cmy_enable', params: [
        { label: 'Cyan', key: 'hs_c', min: -1, max: 2, step: 0.01 },
        { label: 'Magenta', key: 'hs_m', min: -1, max: 2, step: 0.01 },
        { label: 'Yellow', key: 'hs_y', min: -1, max: 2, step: 0.01 },
      ] },
      { subTitle: 'Hue Contrast', subEnableKey: 'hc_enable', params: [
        { label: 'Red', key: 'hc_r', min: 0, max: 2, step: 0.01 },
      ] },
    ],
  },
  {
    title: 'Detail',
    blocks: [{ params: [] }],
  },
  {
    title: 'Output',
    blocks: [
      { params: [{ label: 'Peak Luminance', key: 'peak_luminance', min: 100, max: 2000, step: 10 }] },
    ],
  },
];

/** A Slider bound to an OpenDRT key via the grading context. */
function DrtRow({ g, p }: { g: Grading; p: Param }) {
  return (
    <Slider
      label={p.label} min={p.min} max={p.max} step={p.step}
      value={g.effective(p.key) as number} defaultValue={g.baseConfig[p.key] as number}
      onChange={(v) => g.setDrt(p.key, v as OpenDrtConfig[typeof p.key])}
    />
  );
}

function SubBlock({ g, block }: { g: Grading; block: Group['blocks'][number] }) {
  const enableKey = block.subEnableKey;
  return (
    <div className="xv-pgroup">
      {block.subTitle && (
        <div className="xv-toggle-row">
          <span className="xv-pgroup__title">{block.subTitle}</span>
          {enableKey && (
            <Toggle
              checked={g.effective(enableKey) as boolean}
              label={block.subTitle}
              onChange={(v) => g.setDrt(enableKey, v as OpenDrtConfig[typeof enableKey])}
            />
          )}
        </div>
      )}
      {block.params.map((p) => <DrtRow key={p.key} g={g} p={p} />)}
    </div>
  );
}

export function AdvancedPanel() {
  const g = useGrading();
  const setOpenPanel = useAppStore((s) => s.setOpenPanel);
  const adv = SECTION_KEYS.advanced!;
  const modified = isSectionModified('advanced', g.overrides, g.preOverrides);
  const sharpen = g.effectivePre('sharpen_amount');

  return (
    <FloatingPanel
      title="Advanced"
      modified={modified}
      onReset={() => g.resetSection(adv.drt, adv.pre)}
      onClose={() => setOpenPanel(null)}
    >
      {GROUPS.map((group) => (
        <Collapsible key={group.title} title={group.title}>
          {group.title === 'Detail' ? (
            <Slider
              label="Sharpening" min={0} max={2} step={0.01}
              value={sharpen} defaultValue={0}
              onChange={(v) => g.setPre('sharpen_amount', v)}
            />
          ) : (
            group.blocks.map((block, i) => <SubBlock key={block.subTitle ?? i} g={g} block={block} />)
          )}
        </Collapsible>
      ))}
    </FloatingPanel>
  );
}
