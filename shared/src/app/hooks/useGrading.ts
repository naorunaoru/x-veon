import { useCallback, useMemo } from 'react';
import { useAppStore } from '@/app/store';
import {
  configFromPreset, configWithOverrides, DEFAULT_PREPROCESS,
  type OpenDrtConfig, type PreProcessConfig,
} from '@/renderer/grading/opendrt-params';
import { estimateColorTemperature, findWbTempForCct, findWbTintForTint } from '@/pipeline/color-temperature';
import type { LookPreset } from '@/lib/types';

const EMPTY_DRT: Partial<OpenDrtConfig> = {};
const EMPTY_PRE: Partial<PreProcessConfig> = {};

// Reused scratch buffers (avoid per-render allocation in shootingInfo).
const _tempWbBuf = new Float32Array(3);
const _tintWbBuf = new Float32Array(3);
const _baseWbBuf = new Float32Array(3);

/** Per-file grading context for the floating panels. Reads the selected file. */
export function useGrading() {
  const file = useAppStore((s) => s.files.find((f) => f.id === s.selectedFileId));
  const setFileLookPreset = useAppStore((s) => s.setFileLookPreset);
  const undoFileLook = useAppStore((s) => s.undoFileLook);
  const setFileOpenDrtOverride = useAppStore((s) => s.setFileOpenDrtOverride);
  const setFilePreProcessOverride = useAppStore((s) => s.setFilePreProcessOverride);
  const clearDrt = useAppStore((s) => s.clearFileOpenDrtOverrides);
  const clearPre = useAppStore((s) => s.clearFilePreProcessOverrides);
  const displayHdr = useAppStore((s) => s.displayHdr);
  const displayHdrHeadroom = useAppStore((s) => s.displayHdrHeadroom);

  const fileId = file?.id ?? null;
  const lookPreset: LookPreset = file?.edit.lookPreset ?? 'default';
  const overrides = file?.edit.openDrtOverrides ?? EMPTY_DRT;
  const preOverrides = file?.edit.preProcessOverrides ?? EMPTY_PRE;
  const resultMeta = file?.result?.metadata;
  const exportData = file?.result?.exportData;

  const hdrHeadroom = displayHdr ? displayHdrHeadroom : undefined;
  const baseConfig = useMemo(() => configFromPreset(lookPreset, hdrHeadroom), [lookPreset, hdrHeadroom]);

  const effectiveConfig = useMemo(() => configWithOverrides(baseConfig, overrides), [baseConfig, overrides]);

  const effective = useCallback(
    <K extends keyof OpenDrtConfig>(key: K): OpenDrtConfig[K] =>
      effectiveConfig[key] as OpenDrtConfig[K],
    [effectiveConfig],
  );
  const effectivePre = useCallback(
    (key: keyof PreProcessConfig): number => preOverrides[key] ?? DEFAULT_PREPROCESS[key],
    [preOverrides],
  );

  const setDrt = useCallback(
    <K extends keyof OpenDrtConfig>(key: K, value: OpenDrtConfig[K]) => {
      if (fileId) setFileOpenDrtOverride(fileId, key, value);
    },
    [fileId, setFileOpenDrtOverride],
  );
  const setPre = useCallback(
    <K extends keyof PreProcessConfig>(key: K, value: PreProcessConfig[K]) => {
      if (fileId) setFilePreProcessOverride(fileId, key, value);
    },
    [fileId, setFilePreProcessOverride],
  );
  const setLook = useCallback(
    (preset: LookPreset) => { if (fileId) setFileLookPreset(fileId, preset); },
    [fileId, setFileLookPreset],
  );

  const undoLook = useCallback(() => {
    if (fileId) undoFileLook(fileId);
  }, [fileId, undoFileLook]);

  const handleExposureChange = useCallback(
    (absEv: number) => {
      if (!fileId || !resultMeta) return;
      setFilePreProcessOverride(fileId, 'exposure', absEv - resultMeta.exposureBias);
    },
    [fileId, resultMeta, setFilePreProcessOverride],
  );
  const handleTempChange = useCallback(
    (cct: number) => {
      if (!fileId || !exportData?.camToXyz) return;
      setFilePreProcessOverride(fileId, 'wb_temp', findWbTempForCct(cct, exportData.wbCoeffs, exportData.camToXyz));
    },
    [fileId, exportData, setFilePreProcessOverride],
  );
  const handleTintChange = useCallback(
    (tintVal: number) => {
      if (!fileId || !exportData?.camToXyz) return;
      setFilePreProcessOverride(fileId, 'wb_tint', findWbTintForTint(tintVal, exportData.wbCoeffs, exportData.camToXyz));
    },
    [fileId, exportData, setFilePreProcessOverride],
  );

  // Display CCT / tint / exposure derived from camera WB + the current overrides.
  const shootingInfo = useMemo(() => {
    if (!resultMeta || !exportData?.camToXyz) return null;
    const wb = exportData.wbCoeffs;
    const camToXyz = exportData.camToXyz;
    const temp = preOverrides.wb_temp ?? 0;
    const tint = preOverrides.wb_tint ?? 0;
    const exposure = preOverrides.exposure ?? 0;

    _baseWbBuf[0] = wb[0]; _baseWbBuf[1] = 1.0; _baseWbBuf[2] = wb[2];
    const { temp: baseTempK, tint: baseTint } = estimateColorTemperature(_baseWbBuf, camToXyz);

    // Decouple CCT and tint: estimate each from its own slider only.
    _tempWbBuf[0] = wb[0] * Math.pow(2, temp); _tempWbBuf[1] = 1.0; _tempWbBuf[2] = wb[2] * Math.pow(2, -temp);
    _tintWbBuf[0] = wb[0]; _tintWbBuf[1] = Math.pow(2, -tint); _tintWbBuf[2] = wb[2];
    const { temp: cct } = estimateColorTemperature(_tempWbBuf, camToXyz);
    const { tint: cctTint } = estimateColorTemperature(_tintWbBuf, camToXyz);

    return {
      baseBias: resultMeta.exposureBias,
      exposureEv: resultMeta.exposureBias + exposure,
      baseTempK, tempK: cct,
      baseTint, tintValue: cctTint,
    };
  }, [resultMeta, exportData, preOverrides]);

  const resetSection = useCallback(
    (drtKeys: (keyof OpenDrtConfig)[], preKeys: (keyof PreProcessConfig)[]) => {
      if (!fileId) return;
      if (drtKeys.length) clearDrt(fileId, drtKeys);
      if (preKeys.length) clearPre(fileId, preKeys);
    },
    [fileId, clearDrt, clearPre],
  );

  return {
    fileId,
    hasResult: file?.status === 'done' && file.editing !== 'view-only',
    lookPreset, setLook, baseConfig, undoLook, canUndoLook: !!file?.lookHistory?.length,
    overrides, preOverrides,
    effective, effectivePre, setDrt, setPre,
    shootingInfo, handleExposureChange, handleTempChange, handleTintChange,
    resetSection,
  };
}
