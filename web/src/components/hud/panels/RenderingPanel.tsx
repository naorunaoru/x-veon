import { useMemo, useState } from 'react';
import { RotateCcw, X } from 'lucide-react';
import { useGrading } from '@/app/hooks/useGrading';
import { useAppStore } from '@/app/store';
import { configWithOverrides, type OpenDrtConfig } from '@/renderer/grading/opendrt-params';
import { SECTION_KEYS, isSectionModified } from '@/renderer/grading/sections';
import { RSection } from './rendering/RSection';
import { TonescaleGraph } from './rendering/TonescaleGraph';
import { ColorWheelControl } from './rendering/ColorWheelControl';
import { WarmthPath } from './rendering/WarmthPath';
import { PurityCurve } from './rendering/PurityCurve';
import { ExpertDrawer } from './rendering/ExpertDrawer';
import { BASE_LOOKS } from './rendering/constants';
import './Panels.css';
import './RenderingPanel.css';

type DrtKey = keyof OpenDrtConfig;

// Override keys owned by each in-panel section — drive its dot + reset scope.
const SECTIONS = {
  tonescale: ['tn_lg', 'tn_con', 'tn_toe', 'tn_sh', 'tn_off',
    'tn_lcon_enable', 'tn_lcon', 'tn_lcon_w', 'tn_hcon_enable', 'tn_hcon', 'tn_hcon_pv', 'tn_hcon_st'] as DrtKey[],
  color: ['brl_enable', 'brl_r', 'brl_g', 'brl_b', 'brl_c', 'brl_m', 'brl_y', 'brl_rng',
    'hs_rgb_enable', 'hs_r', 'hs_g', 'hs_b', 'hs_rgb_rng', 'hs_cmy_enable', 'hs_c', 'hs_m', 'hs_y',
    'hc_enable', 'hc_r', 'pt_r', 'pt_g', 'pt_b', 'pt_rng_low'] as DrtKey[],
  whites: ['cwp', 'cwp_rng'] as DrtKey[],
  purity: ['rs_sa', 'ptm_enable', 'ptm_low', 'pt_rng_high'] as DrtKey[],
};

export function RenderingPanel() {
  const g = useGrading();
  const setOpenPanel = useAppStore((s) => s.setOpenPanel);
  const [expert, setExpert] = useState(false);

  const adv = SECTION_KEYS.advanced!;
  const modified = isSectionModified('advanced', g.overrides, g.preOverrides);

  const cfg = useMemo(
    () => configWithOverrides(g.baseConfig, g.overrides, g.preOverrides),
    [g.baseConfig, g.overrides, g.preOverrides],
  );

  const isMod = (keys: DrtKey[]) => keys.some((k) => k in g.overrides);
  const resetKeys = (keys: DrtKey[]) => g.resetSection(keys, []);

  return (
    <div className="xv-rpanel xv-glass-heavy">
      <header className="xv-rpanel__header">
        <span className="xv-rpanel__title">Rendering</span>
        <span className="xv-rpanel__tag">OpenDRT</span>
        {modified && <span className="xv-rpanel__dot" />}
        <div className="xv-rpanel__actions">
          {modified && (
            <button type="button" className="xv-rpanel__icon" aria-label="Reset to look"
              onClick={() => g.resetSection(adv.drt, adv.pre)}>
              <RotateCcw size={12} />
            </button>
          )}
          <button type="button" className="xv-rpanel__icon" aria-label="Close panel" onClick={() => setOpenPanel(null)}>
            <X size={12} />
          </button>
        </div>
      </header>

      <div className="xv-rpanel__body">
        <div className="xv-baselook">
          <div className="xv-rsection__sublabel">Base look</div>
          <div className="xv-baselook__chips">
            {BASE_LOOKS.map((l) => (
              <button key={l.id} type="button"
                className={`xv-rchip${g.lookPreset === l.id ? ' is-selected' : ''}`}
                onClick={() => g.setLook(l.id)}>
                {l.label}
              </button>
            ))}
          </div>
        </div>

        <RSection title="Tonescale" modified={isMod(SECTIONS.tonescale)} onReset={() => resetKeys(SECTIONS.tonescale)}>
          <TonescaleGraph g={g} cfg={cfg} />
        </RSection>

        <RSection title="Color rendering" modified={isMod(SECTIONS.color)} onReset={() => resetKeys(SECTIONS.color)}>
          <ColorWheelControl g={g} />
        </RSection>

        <RSection title="Whites" modified={isMod(SECTIONS.whites)} onReset={() => resetKeys(SECTIONS.whites)}>
          <WarmthPath g={g} />
        </RSection>

        <RSection title="Purity" modified={isMod(SECTIONS.purity)} onReset={() => resetKeys(SECTIONS.purity)}>
          <PurityCurve g={g} />
        </RSection>

        <ExpertDrawer g={g} open={expert} setOpen={setExpert} />
      </div>
    </div>
  );
}
