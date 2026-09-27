import { useRef, useState } from 'react';
import { useAppStore } from '@/app/store';
import { computeMinimap, minimapDragToPan } from '@/lib/minimap';
import './Minimap.css';

// Must match the .xv-minimap width/height in Minimap.css — the rect/img positions
// are computed in JS pixels against this box.
const BOX_W = 200;
const BOX_H = 132.5;

export function Minimap() {
  const file = useAppStore((s) => s.files.find((f) => f.id === s.selectedFileId));
  const scale = useAppStore((s) => s.viewScale);
  const fitScale = useAppStore((s) => s.viewFitScale);
  const pan = useAppStore((s) => s.viewPan);
  const containerW = useAppStore((s) => s.viewContainerW);
  const containerH = useAppStore((s) => s.viewContainerH);
  const controls = useAppStore((s) => s.viewControls);
  const [dragging, setDragging] = useState(false);
  const dragRef = useRef<{
    pointerId: number; startX: number; startY: number;
    startPanX: number; startPanY: number; moved: boolean;
  } | null>(null);

  const result = file?.result;
  const zoomedIn = scale > fitScale * 1.01;
  if (!result || !file?.thumbnailUrl || !containerW || !containerH || !zoomedIn) return null;

  const contentW = result.metadata.width;
  const contentH = result.metadata.height;
  const m = computeMinimap({
    scale, panX: pan.x, panY: pan.y, contentW, contentH, containerW, containerH, boxW: BOX_W, boxH: BOX_H,
  });

  const onPointerDown = (e: React.PointerEvent) => {
    if (!controls || e.button !== 0 || dragRef.current) return;
    e.preventDefault();
    setDragging(true);
    e.currentTarget.setPointerCapture(e.pointerId);
    dragRef.current = {
      pointerId: e.pointerId, startX: e.clientX, startY: e.clientY,
      startPanX: pan.x, startPanY: pan.y, moved: false,
    };
  };
  const onPointerMove = (e: React.PointerEvent) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== e.pointerId || !controls) return;
    const dxBox = e.clientX - drag.startX;
    const dyBox = e.clientY - drag.startY;
    // Allow a little pointer jitter without turning a click into a drag.
    drag.moved ||= Math.hypot(dxBox, dyBox) > 3;
    if (!drag.moved) return;
    const next = minimapDragToPan({
      scale, startPanX: drag.startPanX, startPanY: drag.startPanY,
      dxBox, dyBox,
      contentW, contentH, boxW: BOX_W, boxH: BOX_H,
    });
    controls.panTo(next);
  };
  const onPointerUp = (e: React.PointerEvent) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== e.pointerId) return;
    if (!drag.moved && controls) {
      const bounds = e.currentTarget.getBoundingClientRect();
      // Remove the thumbnail's letterboxing and center the chosen image point.
      const imageX = Math.max(0, Math.min(contentW, (e.clientX - bounds.left - m.imgX) / m.k));
      const imageY = Math.max(0, Math.min(contentH, (e.clientY - bounds.top - m.imgY) / m.k));
      controls.panTo({ x: containerW / 2 - imageX * scale, y: containerH / 2 - imageY * scale });
    }
    dragRef.current = null;
    setDragging(false);
    e.currentTarget.releasePointerCapture(e.pointerId);
  };
  const onPointerCancel = (e: React.PointerEvent) => {
    if (dragRef.current?.pointerId !== e.pointerId) return;
    dragRef.current = null;
    setDragging(false);
  };

  return (
    <div
      className={`xv-minimap xv-glass${dragging ? ' is-dragging' : ''}`}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerCancel}
      onLostPointerCapture={onPointerCancel}
      style={{ touchAction: 'none' }}
    >
      <img
        className="xv-minimap__img" src={file.thumbnailUrl} alt="" draggable={false}
        style={{ left: m.imgX, top: m.imgY, width: m.imgW, height: m.imgH }}
      />
      <div
        className="xv-minimap__rect"
        style={{ left: m.rectX, top: m.rectY, width: m.rectW, height: m.rectH }}
      />
    </div>
  );
}
