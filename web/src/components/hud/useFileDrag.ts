import { useEffect, useRef, useState } from 'react';

/**
 * Window-level file-drag detection. Returns `dragging` (true while a
 * file-bearing drag is anywhere over the window) and fires `onDropFiles` with
 * the dropped `File[]` on drop.
 *
 * A depth counter cancels the noise from dragenter/dragleave bubbling across
 * child elements, so the flag only flips off when the cursor truly leaves the
 * window. We only react when the drag carries files (ignoring text/element
 * drags), and always preventDefault() on dragover/drop so the browser doesn't
 * navigate away to the dropped file.
 */
export function useFileDrag(onDropFiles: (files: File[]) => void): boolean {
  const [dragging, setDragging] = useState(false);
  // Keep the latest callback without re-binding listeners on every render.
  const onDrop = useRef(onDropFiles);
  onDrop.current = onDropFiles;

  useEffect(() => {
    let depth = 0;
    const hasFiles = (e: DragEvent) =>
      Array.from(e.dataTransfer?.types ?? []).includes('Files');

    const onEnter = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      depth++;
      setDragging(true);
    };
    const onOver = (e: DragEvent) => {
      if (hasFiles(e)) e.preventDefault();
    };
    const onLeave = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      depth = Math.max(0, depth - 1);
      if (depth === 0) setDragging(false);
    };
    const onDropEvent = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      depth = 0;
      setDragging(false);
      onDrop.current(Array.from(e.dataTransfer?.files ?? []));
    };

    window.addEventListener('dragenter', onEnter);
    window.addEventListener('dragover', onOver);
    window.addEventListener('dragleave', onLeave);
    window.addEventListener('drop', onDropEvent);
    return () => {
      window.removeEventListener('dragenter', onEnter);
      window.removeEventListener('dragover', onOver);
      window.removeEventListener('dragleave', onLeave);
      window.removeEventListener('drop', onDropEvent);
    };
  }, []);

  return dragging;
}
