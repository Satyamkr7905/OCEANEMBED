"use client";

import type { OceanProfile } from "@/types/ocean";

function tchpStatus(tchp: number | undefined) {
  if (tchp == null) return { label: "NO SOUNDING", tone: "bg-white/10 text-white/70" };
  if (tchp > 80) return { label: "CRITICAL FUEL", tone: "bg-rose-500 text-white" };
  if (tchp >= 50) return { label: "MODERATE FUEL", tone: "bg-amber-400 text-slate-950" };
  return { label: "COLD WAKE", tone: "bg-cyan-400/20 text-cyan-100" };
}

function mldStatus(mld: number | undefined) {
  if (mld == null) return null;
  if (mld < 30) return { label: "SHALLOW MLD", tone: "bg-amber-400/20 text-amber-100" };
  if (mld > 60) return { label: "DEEP MLD", tone: "bg-sky-400/20 text-sky-100" };
  return { label: `MLD ${mld.toFixed(0)} m`, tone: "bg-teal-400/20 text-teal-100" };
}

export default function MetricCards({
  profile,
  latencyMs,
  onGenerateAdvisory,
  generating = false,
}: {
  profile: OceanProfile | null;
  latencyMs: number;
  onGenerateAdvisory?: () => void;
  generating?: boolean;
}) {
  const fuel = tchpStatus(profile?.tchp);
  const mld = mldStatus(profile?.mld);

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
        <MetricTile label="Ocean Heat" value={profile ? profile.tchp.toFixed(1) : "—"} unit="kJ/cm²" hint="Tropical Cyclone Heat Potential" accent="text-orange-300" badge={fuel} />
        <MetricTile label="Mixed Layer" value={profile ? profile.mld.toFixed(1) : "—"} unit="m" hint="Density criterion Δρ = 0.03 kg/m³" accent="text-teal-300" badge={mld ?? undefined} />
        <MetricTile label="Warm Layer" value={profile ? profile.z20.toFixed(1) : "—"} unit="m" hint="Depth of the 20 °C isotherm" accent="text-sky-300" />
        <MetricTile label="Barrier Layer" value={profile?.blt != null ? profile.blt.toFixed(1) : "—"} unit="m" hint="ILD − MLD freshwater cap" accent="text-violet-300" />
        <MetricTile label="Response" value={latencyMs.toFixed(0)} unit="ms" hint="Inference latency" accent="text-cyan-200" />
      </div>

      <div className="glass flex flex-wrap items-center justify-between gap-3 rounded-2xl px-5 py-3.5">
        <div className="flex min-w-0 flex-1 flex-wrap items-center gap-2">
          <span className="font-mono text-[10px] font-bold uppercase tracking-[0.18em] text-cyan-300">
            INCOIS Duty Desk
          </span>
          {profile?.inversion ? (
            <span className="rounded-full bg-amber-400 px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wide text-amber-950">
              Inversion detected
            </span>
          ) : null}
          <span className={`rounded-full px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wide ${fuel.tone}`}>
            {fuel.label}
          </span>
          {mld ? (
            <span className="rounded-full bg-white/10 px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wide text-white/80">
              {mld.label}
            </span>
          ) : null}
        </div>
        {onGenerateAdvisory && (
          <button
            type="button"
            onClick={onGenerateAdvisory}
            disabled={!profile || generating}
            className="shrink-0 rounded-full bg-cyan-400 px-4 py-2.5 text-sm font-semibold text-slate-950 shadow-[0_0_22px_rgba(34,211,238,0.35)] transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {generating ? "Analyzing Subsurface Heat..." : "Generate INCOIS Advisory Dispatch"}
          </button>
        )}
      </div>
    </div>
  );
}

function MetricTile({
  label,
  value,
  unit,
  hint,
  accent,
  badge,
}: {
  label: string;
  value: string;
  unit: string;
  hint: string;
  accent: string;
  badge?: { label: string; tone: string };
}) {
  return (
    <div className="glass rounded-2xl p-4">
      <div className="flex items-start justify-between gap-2">
        <p className={`text-sm font-semibold ${accent}`}>{label}</p>
        {badge ? (
          <span className={`rounded-full px-2 py-0.5 text-[9px] font-bold uppercase tracking-wide ${badge.tone}`}>
            {badge.label}
          </span>
        ) : null}
      </div>
      <p className="mt-1.5 font-mono text-2xl font-bold text-white">
        {value}
        <span className="ml-1 text-xs font-medium text-white/45">{unit}</span>
      </p>
      <p className="mt-1 text-[11px] text-white/45">{hint}</p>
    </div>
  );
}
