"use client";

import type { MapLayerId } from "@/types/ocean";

const GROUPS: { title: string; items: { id: MapLayerId; label: string; fullName: string; color: string }[] }[] = [
  {
    title: "Reconstructed water column",
    items: [
      { id: "theta", label: "Temp", fullName: "Potential temperature", color: "bg-sky-300" },
      { id: "sp", label: "Salt", fullName: "Practical salinity", color: "bg-teal-300" },
      { id: "sigma_theta", label: "Uncertainty", fullName: "Temperature uncertainty", color: "bg-violet-300" },
    ],
  },
  {
    title: "Derived indices",
    items: [
      { id: "tchp", label: "Heat", fullName: "Cyclone heat potential", color: "bg-rose-300" },
      { id: "mld", label: "Mixed", fullName: "Mixed layer depth", color: "bg-sky-200" },
      { id: "z20", label: "Warm", fullName: "Warm layer depth", color: "bg-indigo-300" },
      { id: "blt", label: "Barrier", fullName: "Barrier layer", color: "bg-violet-200" },
      { id: "cip", label: "Cyclone", fullName: "Intensification potential", color: "bg-rose-200" },
    ],
  },
];

export function isIndexLayer(id: MapLayerId): boolean {
  return id === "tchp" || id === "mld" || id === "z20" || id === "blt" || id === "cip";
}

export default function VariableSelector({
  value,
  onChange,
}: {
  value: MapLayerId;
  onChange: (id: MapLayerId) => void;
}) {
  return (
    <div className="space-y-4">
      {GROUPS.map((g) => (
        <div key={g.title}>
          <p className="mb-2 text-xs font-semibold text-white/45">{g.title}</p>
          <div className="flex flex-wrap gap-2">
            {g.items.map((item) => {
              const active = value === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => onChange(item.id)}
                  title={item.fullName}
                  className={`flex items-center gap-2 rounded-full border px-4 py-2 text-sm font-medium transition-all ${
                    active
                      ? "border-cyan-300 bg-cyan-400 text-slate-950 shadow-[0_0_18px_rgba(34,211,238,0.35)]"
                      : "border-white/15 bg-white/5 text-white/70 hover:border-cyan-300/50 hover:text-white"
                  }`}
                >
                  <span className={`inline-block h-2.5 w-2.5 rounded-full ${active ? "bg-slate-950/70" : item.color}`} />
                  {item.label}
                </button>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
