import { test, expect, _electron as electron } from '@playwright/test';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { probeFromReadings } from '../src/host/display';

const samples = ['DSCF3332.RAF', 'sony_a6400_21.arw'];

test('open a folder, process RAF and ARW, edit, export an AVIF that decodes', async ({}, info) => {
  const executable = process.env.XV_SMOKE_APP;
  const source = process.env.XV_SMOKE_SAMPLES;
  if (!executable || !source || !path.isAbsolute(executable) || !path.isAbsolute(source)) {
    throw new Error('XV_SMOKE_APP and XV_SMOKE_SAMPLES must be absolute paths');
  }
  await fs.access(executable, fs.constants.X_OK);
  for (const name of samples) {
    if (!(await fs.stat(path.join(source, name))).isFile()) throw new Error(`Missing sample ${name}`);
  }
  const work = await fs.mkdtemp(path.join(os.tmpdir(), 'xveon-smoke-'));
  const photos = path.join(work, 'photos'), exports = path.join(work, 'exports');
  const target = path.join(exports, 'sony_a6400_21.avif');
  let app: Awaited<ReturnType<typeof electron.launch>> | undefined;
  let tracing = false;
  let exited = false;
  const provenance: Record<string, unknown> = { executable, source, work, photos, target, profile: path.join(work, 'profile') };
  try {
    await fs.mkdir(photos); await fs.mkdir(exports);
    for (const name of samples) await fs.copyFile(path.join(source, name), path.join(photos, name));
    app = await electron.launch({ executablePath: executable, args: [`--user-data-dir=${path.join(work, 'profile')}`], timeout: 30_000 });
    const child = app.process();
    provenance.pid = child.pid;
    child.once('exit', () => { exited = true; });
    await app.context().tracing.start({ screenshots: true, snapshots: true, sources: true });
    tracing = true;
    await app.evaluate(({ dialog, shell }, paths) => {
      dialog.showOpenDialog = (async () => ({ canceled: false, filePaths: [paths.photos] })) as typeof dialog.showOpenDialog;
      dialog.showSaveDialog = (async () => ({ canceled: false, filePath: paths.target })) as typeof dialog.showSaveDialog;
      dialog.showMessageBox = (async () => ({ response: 0, checkboxChecked: false })) as typeof dialog.showMessageBox;
      (globalThis as { revealed?: string[] }).revealed = [];
      shell.showItemInFolder = (file: string) => { (globalThis as { revealed?: string[] }).revealed!.push(file); };
    }, { photos, target });
    const page = await app.firstWindow();
    page.setDefaultTimeout(30_000);
    await page.getByRole('button', { name: 'Open folder…', exact: true }).click();
    await page.getByRole('menuitem', { name: 'Open folder…', exact: true }).click();
    const thumbs = page.getByTestId('filmstrip-thumb');
    await expect(thumbs).toHaveCount(2, { timeout: 60_000 });
    await expect(thumbs.nth(0).locator('.xv-thumb__dot.done')).toBeVisible({ timeout: 180_000 });
    await thumbs.nth(1).click();
    await expect(thumbs.nth(1).locator('.xv-thumb__dot.done')).toBeVisible({ timeout: 180_000 });
    const readings = await page.evaluate(() => (window as unknown as { xveon: { displayReadings(): Promise<unknown> } }).xveon.displayReadings());
    provenance.displayReadings = readings;
    provenance.appName = await app.evaluate(({ app }) => app.getName());
    expect(readings, 'native display readings in the packaged app').not.toBeNull();
    const expected = probeFromReadings(readings as Parameters<typeof probeFromReadings>[0])!;
    await page.getByRole('button', { name: 'Settings', exact: true }).click();
    const output = page.locator('.xv-readout').filter({ hasText: 'HDR preview' });
    await expect(output).toContainText(expected.supported ? `peak ${Math.round(expected.headroom * 100)} nits` : 'OFF');
    await page.getByRole('button', { name: 'Settings', exact: true }).click();
    await page.getByRole('button', { name: 'Exposure', exact: true }).click();
    await page.getByRole('slider', { name: 'Exposure', exact: true }).focus();
    await page.keyboard.press('ArrowRight');
    await expect.poll(() => fs.readFile(path.join(photos, 'sony_a6400_21.arw.xmp'), 'utf8').catch(() => ''), { timeout: 30_000 }).toContain('xveon:Exposure="0.01"');
    await page.getByRole('button', { name: 'Export', exact: true }).click();
    const dialog = page.getByRole('dialog');
    const avif = dialog.getByRole('radio', { name: 'AVIF (BT.2020 / HLG)', exact: true });
    await avif.focus();
    await page.keyboard.press('Space');
    await expect(avif).toBeChecked();
    await dialog.getByRole('button', { name: 'Export', exact: true }).click();
    await expect(page.locator('.xv-export-status[role="status"]')).toContainText('Exported sony_a6400_21.avif', { timeout: 120_000 });
    await expect(page.getByLabel('Adjustments', { exact: true })).toBeVisible();
    await page.getByRole('button', { name: /Show in (Finder|Explorer)/ }).click();
    await expect(page.getByLabel('Adjustments', { exact: true })).toBeVisible();
    expect(await app.evaluate(() => (globalThis as { revealed?: string[] }).revealed)).toEqual([target]);
    const size = await app.evaluate(async ({ BrowserWindow }, url) => {
      const probe = new BrowserWindow({ show: false });
      try {
        await probe.loadURL(url);
        return await probe.webContents.executeJavaScript(`new Promise((resolve, reject) => {
          const image = new Image();
          const timer = setTimeout(() => reject(new Error('AVIF decode timed out')), 15000);
          image.onload = () => { clearTimeout(timer); resolve([image.naturalWidth, image.naturalHeight]); };
          image.onerror = () => { clearTimeout(timer); reject(new Error('AVIF decode failed')); };
          image.src = ${JSON.stringify(url)};
          document.body.append(image);
        })`);
      } finally { probe.destroy(); }
    }, pathToFileURL(target).href);
    expect(size).toEqual([6024, 4024]);
  } finally {
    if (app) {
      if (tracing) {
        const trace = info.outputPath('electron-trace.zip');
        await app.context().tracing.stop({ path: trace }).catch(error => { provenance.traceError = String(error); });
        if (info.status !== info.expectedStatus) await info.attach('trace', { path: trace, contentType: 'application/zip' }).catch(() => {});
        else await fs.rm(trace, { force: true });
      }
      // Bound the native close guard; kill only the process returned by this launch.
      const child = app.process();
      let timer: ReturnType<typeof setTimeout> | undefined;
      await Promise.race([app.close().catch(() => {}), new Promise<void>(resolve => { timer = setTimeout(resolve, 10_000); })]);
      clearTimeout(timer);
      if (!exited && child.exitCode === null && child.signalCode === null) child.kill('SIGKILL');
      await new Promise<void>(resolve => {
        if (exited || child.exitCode !== null || child.signalCode !== null) { exited = true; resolve(); return; }
        const deadline = setTimeout(resolve, 5_000);
        child.once('exit', () => { clearTimeout(deadline); exited = true; resolve(); });
      });
    } else exited = true;
    provenance.confirmedExit = exited;
    if (exited) await fs.rm(work, { recursive: true, force: true });
    provenance.cleaned = exited;
    await fs.writeFile(info.outputPath('provenance.json'), JSON.stringify(provenance, null, 2));
    await info.attach('provenance', { path: info.outputPath('provenance.json'), contentType: 'application/json' });
    expect(exited, 'Owned Electron process must exit before cleanup').toBe(true);
  }
});
