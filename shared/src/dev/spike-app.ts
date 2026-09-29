import { useAppStore } from '@/app/store';
import { importFiles } from '@/app/services/library';
import { SAMPLE_CONTRACT } from './golden-contract';
export async function waitForSpikeApp() {
  const deadline = performance.now() + 120_000;
  while (!useAppStore.getState().initialized) {
    const error = useAppStore.getState().initError;
    if (error) throw Error(error);
    if (performance.now() > deadline)
      throw Error('App initialization timed out');
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
}
/** Leave the same fixture on screen in each runtime for the physical HDR comparison. */
export async function showSpikeFixture() {
  await waitForSpikeApp();
  const selected =
    new URLSearchParams(location.search).get('sample') ?? 'xtrans';
  const sample = SAMPLE_CONTRACT.find((item) => item.cfa === selected);
  if (!sample) throw Error('Unknown spike fixture');
  useAppStore.getState().setDemosaicMethod('neural-net');
  useAppStore.getState().setModelSize('S');
  const response = await fetch(
    `${import.meta.env.BASE_URL}samples/${sample.file}`,
  );
  if (!response.ok) throw Error('Missing spike fixture');
  await importFiles([new File([await response.arrayBuffer()], sample.file)]);
}
