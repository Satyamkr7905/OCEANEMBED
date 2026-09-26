"use client";

import { NIO } from "@/types/ocean";

export default function MapControls({
  opacity,
  onOpacity,
  onBasin,
}: {
  opacity: number;
  onOpacity: (v: number) => void;
  onBasin: (b: "nio" | "as" | "bob") => void;
}) {
  return (
    <div className="pointer-events-auto w-56 space-y-3 rounded-2xl border border-white/12 bg-[#061422]/78 p-4 shadow-lifted backdrop-blur-xl">
      <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-cyan-200/70">Basin focus</p>
      <div className="flex gap-1.5">
        {(
          [
            ["nio", "NIO"],
            ["as", "Arabian"],
            ["bob", "BoB"],
          ] as const
        ).map(([id, label]) => (
          <button
            key={id}
            type="button"
            onClick={() => onBasin(id)}
            className="flex-1 rounded-lg border border-white/15 bg-white/5 py-1.5 text-center font-mono text-[10px] font-medium text-white/70 transition-all hover:border-cyan-300 hover:bg-cyan-400/15 hover:text-cyan-100"
          >
            {label}
          </button>
        ))}
      </div>
      <div>
        <div className="mb-1.5 flex items-center justify-between">
          <span className="text-[10px] font-bold uppercase tracking-widest text-white/40">Opacity</span>
          <span className="font-mono text-[11px] font-semibold text-cyan-200">{Math.round(opacity * 100)}%</span>
        </div>
        <input
          type="range"
          min={0.15}
          max={1}
          step={0.05}
          value={opacity}
          onChange={(e) => onOpacity(Number(e.target.value))}
          className="w-full"
        />
      </div>
      <p className="font-mono text-[9px] text-white/40">
        {NIO.latMin}–{NIO.latMax}°N · {NIO.lonMin}–{NIO.lonMax}°E
      </p>
    </div>
  );
}
