import { BUILD } from '@/lib/channel';
import { getPipeline } from '@/app/services/processing';
import { waitForSpikeApp } from './spike-app';
import { processRaw } from '@/pipeline';
import { SAMPLE_CONTRACT } from './golden-contract';
/** Same pipeline and timing boundary on each host; file IO and model load reported separately. */
export async function runTiming() {
  const initStart = performance.now();
  await waitForSpikeApp();
  const ctx = getPipeline();
  const initializationWaitMs = performance.now() - initStart;
  const fixtures = [];
  const selected = new URLSearchParams(location.search).get('sample');
  for (const sample of SAMPLE_CONTRACT.filter(
    (s) => !selected || s.cfa === selected,
  )) {
    const raw = await fetch(
      `${import.meta.env.BASE_URL}samples/${sample.file}`,
    );
    if (!raw.ok) throw Error('Missing fixture');
    const bytes = await raw.arrayBuffer();
    const runs = [];
    for (let index = 0; index < 7; index++) {
      const timings: { demosaicMs?: number } = {};
      const start = performance.now();
      const result = await processRaw(
        bytes.slice(0),
        { method: 'neural-net', modelSize: 'S', timings },
        ctx,
      );
      await ctx.device.queue.onSubmittedWorkDone();
      const row = {
        phase: index === 0 ? 'first' : index === 1 ? 'warmup' : 'measured',
        processingMs: performance.now() - start,
        inferenceMs: timings.demosaicMs!,
        demosaicPostprocessMs: result.meta.metadata.inferenceTime * 1000,
        metadata: result.meta.metadata,
      };
      if (row.metadata.backend !== 'webgpu')
        throw Error('Timing requires WebGPU inference');
      result.dispose();
      runs.push(row);
    }
    fixtures.push({ file: sample.file, runs });
  }
  if (!fixtures.length) throw Error('Unknown timing sample');
  const a = ctx.device.adapterInfo;
  return {
    commit: BUILD.sha,
    recordedAt: new Date().toISOString(),
    isSecureContext,
    crossOriginIsolated,
    hdrMedia: matchMedia('(dynamic-range: high)').matches,
    initializationWaitMs,
    adapter: {
      vendor: a.vendor,
      architecture: a.architecture,
      device: a.device,
      description: a.description,
    },
    fixtures,
  };
}
