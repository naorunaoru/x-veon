import { useAppStore } from '@/store';
import './FileMetaPill.css';

/** Strip Fujifilm's redundant make prefix from the camera display. */
function cleanCamera(camera: string): string {
  return camera.replace(/^fujifilm\s+/i, '').trim();
}

export function FileMetaPill() {
  const file = useAppStore((s) => s.files.find((f) => f.id === s.selectedFileId));
  if (!file) return null;

  const m = file.metadata;

  // Camera gets its own part so it appears immediately (available from RAF header).
  const camera = cleanCamera(m?.camera ?? '');

  // Lens model, focal length, and aperture are decoded later; group them in one
  // part span so getByText regex matching works even when the lens name contains
  // digits that overlap with the focal-length pattern (e.g. "XF35mmF1.4 R").
  const lensSubParts: string[] = [];
  if (m?.lensModel) lensSubParts.push(m.lensModel);
  if (m && m.focalLength > 0) lensSubParts.push(`${Math.round(m.focalLength)}mm`);
  if (m && m.fNumber > 0) lensSubParts.push(`f/${parseFloat(m.fNumber.toFixed(1))}`);
  const lensGroup = lensSubParts.join(' · ');

  const parts = [camera, lensGroup].filter(Boolean);

  return (
    <div className="xv-filemeta xv-glass xv-over-image">
      <span className="xv-filemeta__name">{file.originalName}</span>
      {parts.map((p, i) => (
        <span key={i} className="xv-filemeta__part">
          <span className="xv-filemeta__sep">·</span> {p}
        </span>
      ))}
    </div>
  );
}
