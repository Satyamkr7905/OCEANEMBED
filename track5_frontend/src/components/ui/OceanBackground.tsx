"use client";

import { useEffect, useRef, useState } from "react";

const FRAME_COUNT = 45;
const LOOP_MS = 90;

type FrameImage = ImageBitmap | HTMLImageElement;

function frameSrc(index: number): string {
  return `/frames/frame_${String(index + 1).padStart(4, "0")}.webp`;
}

function coverRect(imgW: number, imgH: number, viewW: number, viewH: number) {
  const scale = Math.max(viewW / imgW, viewH / imgH);
  return { x: (viewW - imgW * scale) / 2, y: (viewH - imgH * scale) / 2, w: imgW * scale, h: imgH * scale };
}

export default function OceanBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frames = useRef<Array<FrameImage | null>>(Array(FRAME_COUNT).fill(null));
  const [loaded, setLoaded] = useState(0);

  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      for (let i = 0; i < FRAME_COUNT; i += 1) {
        try {
          const blob = await (await fetch(frameSrc(i))).blob();
          frames.current[i] = "createImageBitmap" in window
            ? await createImageBitmap(blob)
            : await new Promise<HTMLImageElement>((resolve, reject) => {
                const img = new Image();
                img.onload = () => resolve(img);
                img.onerror = () => reject(new Error("frame"));
                img.src = URL.createObjectURL(blob);
              });
        } catch {
          frames.current[i] = null;
        }
        if (!cancelled) setLoaded(i + 1);
      }
    };
    void run();
    return () => {
      cancelled = true;
      frames.current.forEach((im) => {
        if (im && "close" in im && typeof im.close === "function") im.close();
      });
    };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || loaded === 0) return;
    let raf = 0;
    let last = 0;
    let idx = 0;
    const tick = (now: number) => {
      if (now - last >= LOOP_MS) {
        last = now;
        idx = (idx + 1) % FRAME_COUNT;
      }
      const img = frames.current[idx];
      const ctx = canvas.getContext("2d");
      if (ctx && img) {
        const dpr = Math.min(2, window.devicePixelRatio || 1);
        const w = canvas.clientWidth;
        const h = canvas.clientHeight;
        if (canvas.width !== Math.floor(w * dpr) || canvas.height !== Math.floor(h * dpr)) {
          canvas.width = Math.floor(w * dpr);
          canvas.height = Math.floor(h * dpr);
        }
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        const iw = "width" in img ? img.width : (img as HTMLImageElement).naturalWidth;
        const ih = "height" in img ? img.height : (img as HTMLImageElement).naturalHeight;
        const r = coverRect(iw, ih, w, h);
        ctx.drawImage(img as CanvasImageSource, r.x, r.y, r.w, r.h);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [loaded]);

  return (
    <div className="pointer-events-none fixed inset-0 z-0">
      <canvas ref={canvasRef} className="h-full w-full" aria-hidden />
      <div className="absolute inset-0 bg-gradient-to-b from-black/40 via-black/25 to-black/50" />
    </div>
  );
}
