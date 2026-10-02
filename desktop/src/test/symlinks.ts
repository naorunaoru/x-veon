import { inject, type TestContext } from 'vitest';
export function requireFileSymlinks(context: TestContext) {
 const reason = inject('fileSymlinkSkipReason');
 if (reason) context.skip(reason);
}
export const directoryLinkType = process.platform === 'win32' ? 'junction' : 'dir';
