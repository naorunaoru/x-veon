import { createHmac } from 'node:crypto';
import path from 'node:path';

export function photoId(sessionKey: Buffer, absolutePath: string): string {
  return createHmac('sha256', sessionKey).update(path.resolve(absolutePath)).digest('base64url').slice(0, 22);
}
