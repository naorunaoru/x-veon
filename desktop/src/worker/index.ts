import { createExportService } from './exports';
import { loadNativeEncoder } from './native';
import type {} from 'electron';
import { createWorkerController } from './controller';
import { createWatchHandler } from './watch';
const controller = createWorkerController({
  exports: createExportService({ native: () => loadNativeEncoder() }),
  postToMain: message => process.parentPort.postMessage(message),
  onWatch: folder => watch(folder),
});
const watch = createWatchHandler(() => controller.replace());
process.parentPort.on('message', event => {
  void controller.handleMain(event.data, event.ports).catch(error => {
    // Invalid control/session state is fatal; the supervisor restores a fresh worker.
    console.error(error); process.exit(1);
  });
});
