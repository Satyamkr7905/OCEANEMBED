"use client";

import { DEPTHS, type DepthM } from "@/types/ocean";

const DEPTH_ZONES = [
  { label: "Surface", range: "0–20 m", max: 20 },
  { label: "Mixed layer", range: "20–100 m", max: 100 },
  { label: "Thermocline", range: "100–500 m", max: 500 },
  { label: "Deep", range: "500–1000 m", max: 1000 },
] as const;

export default function DepthSlider({ value, onChange }: { value: DepthM; onChange: (z: DepthM) => void }) {
  const idx = DEPTHS.indexOf(value);
  const zone = DEPTH_ZONES.find((z) => value <= z.max) ?? DEPTH_ZONES[3];
  const pct = ((idx < 0 ? 0 : idx) / (DEPTHS.length - 1)) * 100;

  return (
    <div>
      <div className="mb-3 flex items-baseline justify-between">
        <span className="text-sm font-semibold text-white/70">Depth</span>
        <div className="text-right">
          <span className="font-mono text-2xl font-bold text-cyan-300">{value}</span>
          <span className="ml-1 text-sm text-white/40">m</span>
        </div>
      </div>
      <div className="relative">
        <div className="absolute left-0 top-1/2 h-2 -translate-y-1/2 rounded-full bg-cyan-400 transition-all" style={{ width: `${pct}%` }} />
        <input
          type="range"
          min={0}
          max={DEPTHS.length - 1}
          step={1}
          value={idx < 0 ? 0 : idx}
          onChange={(e) => onChange(DEPTHS[Number(e.target.value)])}
          className="relative z-10 w-full"
        />
      </div>
      <div className="mt-2 flex justify-between text-xs text-white/40">
        <span>0 m</span>
        <span className="rounded-full bg-white/10 px-3 py-1 text-xs font-medium text-cyan-100">{zone.label}</span>
        <span>1000 m</span>
      </div>
    </div>
  );
}
