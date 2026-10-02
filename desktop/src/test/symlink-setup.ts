import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import type { TestProject } from 'vitest/node';

// One capability probe per Vitest invocation. Only Windows privilege denial skips.
export default async function setup(project: TestProject) {
 const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'xveon-link-probe-'));
 let reason: string | null = null;
 try {
   const target = path.join(dir, 'target'); await fs.writeFile(target, 'probe');
   await fs.symlink(target, path.join(dir, 'link'), 'file');
 } catch (error) {
   if (process.platform !== 'win32' || (error as NodeJS.ErrnoException).code !== 'EPERM') throw error;
   reason = 'Windows file symlink privilege unavailable (EPERM); file-link cases only are skipped.';
   console.warn(reason);
 } finally { await fs.rm(dir, { recursive: true, force: true }); }
 project.provide('fileSymlinkSkipReason', reason);
}
declare module 'vitest' { export interface ProvidedContext { fileSymlinkSkipReason: string | null } }
