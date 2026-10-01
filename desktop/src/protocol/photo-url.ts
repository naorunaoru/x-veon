import type { PhotoId } from '@/host';
export function parsePhotoUrl(raw: string): { kind: 'raw' | 'thumb'; id: PhotoId } | null {
  const match = /^xveon-photo:\/\/(raw|thumb)\/([A-Za-z0-9_-]{22})$/.exec(raw);
  return match ? { kind: match[1] as 'raw' | 'thumb', id: match[2] } : null;
}
export function photoUrl(kind: 'raw' | 'thumb', id: PhotoId): string {
  const url = `xveon-photo://${kind}/${id}`;
  if (!parsePhotoUrl(url)) throw new Error('Invalid photo URL');
  return url;
}
