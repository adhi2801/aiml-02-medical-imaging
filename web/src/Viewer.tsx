import { useCallback, useEffect, useRef, useState } from "react";

interface Props {
  imageUrl: string;
  camUrl: string;
  showOverlay: boolean;
  opacity: number;
}

interface View {
  zoom: number;
  x: number;
  y: number;
  brightness: number; // window level, 1 = as acquired
  contrast: number; // window width
}

const RESET: View = { zoom: 1, x: 0, y: 0, brightness: 1, contrast: 1 };

/**
 * Scan + Grad-CAM viewport.
 *   drag            pan
 *   shift+drag      window / level (horizontal = contrast, vertical = brightness)
 *   wheel           zoom at cursor
 *   double-click    reset
 */
export function Viewer({ imageUrl, camUrl, showOverlay, opacity }: Props) {
  const [view, setView] = useState<View>(RESET);
  const drag = useRef<{ x: number; y: number; wl: boolean } | null>(null);
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => setView(RESET), [imageUrl]);

  const onWheel = useCallback((e: WheelEvent) => {
    e.preventDefault();
    const rect = box.current!.getBoundingClientRect();
    const cx = e.clientX - rect.left - rect.width / 2;
    const cy = e.clientY - rect.top - rect.height / 2;
    setView((v) => {
      const zoom = Math.min(8, Math.max(1, v.zoom * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
      const k = zoom / v.zoom;
      return zoom === 1 ? { ...v, zoom, x: 0, y: 0 } : { ...v, zoom, x: cx - (cx - v.x) * k, y: cy - (cy - v.y) * k };
    });
  }, []);

  useEffect(() => {
    const el = box.current!;
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [onWheel]);

  const onPointerDown = (e: React.PointerEvent) => {
    (e.target as Element).setPointerCapture(e.pointerId);
    drag.current = { x: e.clientX, y: e.clientY, wl: e.shiftKey };
  };
  const onPointerMove = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    const dx = e.clientX - d.x;
    const dy = e.clientY - d.y;
    d.x = e.clientX;
    d.y = e.clientY;
    setView((v) =>
      d.wl
        ? {
            ...v,
            contrast: Math.min(4, Math.max(0.25, v.contrast * (1 + dx / 300))),
            brightness: Math.min(3, Math.max(0.2, v.brightness * (1 - dy / 300))),
          }
        : { ...v, x: v.x + dx, y: v.y + dy },
    );
  };
  const onPointerUp = () => (drag.current = null);

  const transform = `translate(${view.x}px, ${view.y}px) scale(${view.zoom})`;
  const filter = `brightness(${view.brightness}) contrast(${view.contrast})`;

  return (
    <div
      ref={box}
      className="viewport"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onDoubleClick={() => setView(RESET)}
    >
      <div className="stack" style={{ transform }}>
        <img src={imageUrl} alt="Chest radiograph" draggable={false} style={{ filter }} />
        <img
          src={camUrl}
          alt="Grad-CAM heatmap"
          draggable={false}
          className="cam"
          style={{ opacity: showOverlay ? opacity : 0 }}
        />
      </div>
      <div className="vp-readout mono">
        Z {view.zoom.toFixed(2)}× · L {view.brightness.toFixed(2)} · W {view.contrast.toFixed(2)}
      </div>
      <div className="vp-hint">drag pan · shift+drag window/level · wheel zoom · dbl-click reset</div>
    </div>
  );
}
