import '../protocol/bridge';
import { waitForSpikeApp, showSpikeFixture } from '@/dev/spike-app';
import { runGolden } from '@/dev/golden';
import { useAppStore } from '@/app/store';
import { discardResult } from '@/app/services/processing';
import { runTiming } from '@/dev/spike-timing';
import { getDevice } from '@/gpu/device';
import { runTransport } from './transport';
export async function runSpike(release: (id: string) => void) {
  let unsubscribe = () => {};
  try {
    const params = new URLSearchParams(location.search);
    let canvasConfiguration: unknown = null;
    unsubscribe = useAppStore.subscribe((state) => {
      if (!state.renderer) return;
      const canvas = document.querySelector<HTMLCanvasElement>(
        '.xv-canvas-host canvas',
      );
      const config = canvas?.getContext('webgpu')?.getConfiguration();
      if (config)
        canvasConfiguration = {
          format: config.format,
          colorSpace: config.colorSpace,
          toneMapping: config.toneMapping,
          alphaMode: config.alphaMode,
        };
    });
    await waitForSpikeApp();
    const device = await getDevice();
    const a = device.adapterInfo;
    await window.xveon.report('capabilities', {
      origin: location.origin,
      preloadEnvironment: window.xveon.environment,
      isSecureContext,
      crossOriginIsolated,
      nodeExposed: Object.prototype.hasOwnProperty.call(window, 'process'),
      adapter: {
        vendor: a.vendor,
        architecture: a.architecture,
        device: a.device,
        description: a.description,
      },
      hdrMedia: matchMedia('(dynamic-range: high)').matches,
      diagnostics: await window.xveon.request('diagnostics'),
    });
    if (
      !window.xveon.environment.sandboxed ||
      !window.xveon.environment.contextIsolated ||
      crossOriginIsolated ||
      Object.prototype.hasOwnProperty.call(window, 'process')
    )
      throw Error('Unexpected renderer isolation');
    if (params.has('golden')) {
      await runGolden(async (id) => {
        discardResult(id);
        useAppStore.getState().removeFile(id);
        release(id);
      });
      await window.xveon.report('golden', {
        ...(window as any).__golden,
        canvasConfiguration,
      });
    } else if (params.get('spike') === 'view') {
      await showSpikeFixture();
    } else if (params.get('spike') === 'timing') {
      const result = await runTiming();
      await window.xveon.report('timing', {
        runtime: navigator.userAgent,
        ...result,
      });
      document.title = 'spike timing: complete';
    } else if (params.get('spike') === 'transport') {
      const result = await runTransport();
      await window.xveon.report('transport', result);
      document.title = 'spike transport: PASS';
    }
  } catch (error) {
    document.title = 'spike: ERROR';
    console.error(error);
    await window.xveon.report('error', { error: String(error) });
  } finally {
    unsubscribe();
  }
}
