import { useRef, useState } from 'react';
import { useAppStore } from '@/store';
import { computeMinimap, minimapDragToPan } from '@/lib/minimap';
import './Minimap.css';

const BOX_W = 160;
const BOX_H = 106;

export function Minimap() {
  const file = useAppStore((s) => s.files.find((f) => f.id === s.selectedFileId));
  const scale = useAppStore((s) => s.viewScale);
  const fitScale = useAppStore((s) => s.viewFitScale);
  const pan = useAppStore((s) => s.viewPan);
  const containerW = useAppStore((s) => s.viewContainerW);
  const containerH = useAppStore((s) => s.viewContainerH);
  const controls = useAppStore((s) => s.viewControls);
  const [dragging, setDragging] = useState(false);
  const dragRef = useRef({ startX: 0, startY: 0, startPanX: 0, startPanY: 0 });

  const result = file?.result;
  const zoomedIn = scale > fitScale * 1.01;
  if (!result || !file?.thumbnailUrl || !containerW || !containerH || !zoomedIn) return null;

  const contentW = result.metadata.width;
  const contentH = result.metadata.height;
  const m = computeMinimap({
    scale, panX: pan.x, panY: pan.y, contentW, contentH, containerW, containerH, boxW: BOX_W, boxH: BOX_H,
  });

  const onPointerDown = (e: React.PointerEvent) => {
    setDragging(true);
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    dragRef.current = { startX: e.clientX, startY: e.clientY, startPanX: pan.x, startPanY: pan.y };
  };
  const onPointerMove = (e: React.PointerEvent) => {
    if (!dragging || !controls) return;
    const next = minimapDragToPan({
      scale, startPanX: dragRef.current.startPanX, startPanY: dragRef.current.startPanY,
      dxBox: e.clientX - dragRef.current.startX, dyBox: e.clientY - dragRef.current.startY,
      contentW, contentH, boxW: BOX_W, boxH: BOX_H,
    });
    controls.panTo(next);
  };
  const onPointerUp = (e: React.PointerEvent) => {
    setDragging(false);
    (e.target as HTMLElement).releasePointerCapture(e.pointerId);
  };

  return (
    <div
      className={`xv-minimap xv-glass${dragging ? ' is-dragging' : ''}`}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      style={{ touchAction: 'none' }}
    >
      <img
        className="xv-minimap__img" src={file.thumbnailUrl} alt=""
        style={{ left: m.imgX, top: m.imgY, width: m.imgW, height: m.imgH }}
      />
      <div
        className="xv-minimap__rect"
        style={{ left: m.rectX, top: m.rectY, width: m.rectW, height: m.rectH }}
      />
    </div>
  );
}
