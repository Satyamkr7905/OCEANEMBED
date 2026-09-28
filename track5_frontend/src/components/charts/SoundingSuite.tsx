"use client";

import { useMemo, useState, type ReactNode } from "react";
import ThermalSoundingChart from "@/components/charts/ThermalSoundingChart";
import SalinityDensityChart from "@/components/charts/SalinityDensityChart";
import ThermalGradientHeatChart from "@/components/charts/ThermalGradientHeatChart";
import type { OceanProfile } from "@/types/ocean";
import { derivedSounding } from "@/lib/soundingPhysics";

type SuiteTab = "all" | "thermal" | "halocline" | "heat";

const TABS: Array<{ id: SuiteTab; label: string }> = [
  { id: "all", label: "All 3" },
  { id: "thermal", label: "Thermal" },
  { id: "halocline", label: "Halocline & barrier" },
  { id: "heat", label: "TCHP & gradient" },
];

export default function SoundingSuite({
  profile,
  loading,
  selected,
}: {
  profile: OceanProfile | null;
  loading: boolean;
  selected: { lat: number; lon: number } | null;
}) {
  const [tab, setTab] = useState<SuiteTab>("all");
  const [legendOpen, setLegendOpen] = useState(false);
  const derived = useMemo(() => (profile ? derivedSounding(profile) : null), [profile]);
  const stacked = tab === "all";
  const chartH = stacked ? 500 : 620;

  return (
    <section className="glass rounded-3xl">
      <header className="border-b border-white/10 px-6 py-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="font-mono text-xs font-bold uppercase tracking-[0.2em] text-cyan-300">
              Scientific sounding
            </p>
            <h3 className="mt-1 text-lg font-bold text-white">Vertical diagnostics</h3>
            {selected ? (
              <p className="mt-1 font-mono text-sm text-white/60">
                {selected.lat.toFixed(2)}°N {selected.lon.toFixed(2)}°E
                {profile ? ` · ${profile.date}` : ""}
              </p>
            ) : (
              <p className="mt-1 text-sm text-white/50">Click the map to pin a station</p>
            )}
          </div>
          <button
            type="button"
            onClick={() => setLegendOpen((v) => !v)}
            className="rounded-lg border border-white/15 px-3 py-2 font-mono text-xs font-semibold uppercase tracking-wide text-cyan-100/80 hover:border-cyan-300 hover:text-cyan-200"
          >
            {legendOpen ? "Hide legend" : "Thresholds"}
          </button>
        </div>

        <div className="mt-4 flex flex-wrap gap-1.5 rounded-xl bg-white/5 p-1.5">
          {TABS.map((t) => (
            <button
              key={t.id}
              type="button"
              onClick={() => setTab(t.id)}
              className={`rounded-lg px-4 py-2 font-mono text-xs font-semibold ${
                tab === t.id ? "bg-cyan-400 text-slate-950 shadow-[0_0_18px_rgba(34,211,238,0.28)]" : "text-white/60 hover:text-white"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>

        {loading && (
          <div className="mt-3 h-1 overflow-hidden rounded-full bg-white/10">
            <div className="h-full w-1/3 animate-pulse rounded-full bg-cyan-400" />
          </div>
        )}
      </header>

      {legendOpen && (
        <div className="border-b border-white/10 bg-cyan-400/5 px-6 py-4 font-mono text-xs leading-relaxed text-white/70">
          <p className="font-semibold uppercase tracking-[0.16em] text-cyan-200">Physical thresholds</p>
          <ul className="mt-2.5 grid gap-2 sm:grid-cols-2">
            <li><span className="font-semibold text-amber-300">MLD</span> — density criterion Δρ = 0.03 kg/m³ from surface.</li>
            <li><span className="font-semibold text-sky-300">ILD</span> — isothermal layer, ΔT = 0.5 °C from SST.</li>
            <li><span className="font-semibold text-rose-300">Z₂₀ / D26</span> — 20 °C and 26 °C isotherms. TCHP integrates T−26 above D26.</li>
            <li><span className="font-semibold text-violet-300">BLT</span> — barrier layer = ILD − MLD (fresh cap over warm water).</li>
          </ul>
        </div>
      )}

      <div className="p-5">
        {!profile ? (
          <div className="flex h-[380px] flex-col items-center justify-center text-center">
            <p className="text-base font-semibold text-white/80">No sounding loaded</p>
            <p className="mt-1.5 max-w-sm text-sm text-white/50">
              Pin a location on the North Indian Ocean map. Thermal, salinity and heat diagnostics appear here.
            </p>
          </div>
        ) : (
          <div className={stacked ? "grid gap-5 lg:grid-cols-3" : "grid gap-5"}>
            {(stacked || tab === "thermal") && (
              <ChartCard eyebrow="Chart 1" title="Vertical thermal profile" note="θ, ±1σ, ARGO, MLD / Z₂₀">
                <ThermalSoundingChart profile={profile} height={chartH} />
              </ChartCard>
            )}
            {(stacked || tab === "halocline") && (
              <ChartCard eyebrow="Chart 2" title="Halocline & barrier layer" note="Upper 300 m · Sₚ + σ_θ">
                <SalinityDensityChart profile={profile} height={chartH} />
              </ChartCard>
            )}
            {(stacked || tab === "heat") && (
              <ChartCard
                eyebrow="Chart 3"
                title="Thermocline gradient & cyclone heat"
                note={
                  derived?.thermoclineCore
                    ? `Sharpest dT/dz at ${derived.thermoclineCore.zMid.toFixed(0)} m`
                    : "dT/dz core and layer heat vs 26 °C"
                }
              >
                <ThermalGradientHeatChart profile={profile} height={chartH} />
              </ChartCard>
            )}
          </div>
        )}
      </div>

      {profile && (
        <footer className="flex items-center justify-between gap-3 border-t border-white/10 px-6 py-3.5">
          <div className="flex items-center gap-2.5">
            <span
              className={`h-3 w-3 rounded-full ${
                profile.backend === "pytorch_real"
                  ? "bg-cyan-400"
                  : profile.backend === "amphan_case_study"
                    ? "bg-orange-400"
                    : profile.source === "api"
                      ? "bg-teal-400"
                      : "bg-amber-400"
              }`}
            />
            <span className="font-mono text-xs text-white/60">
              {profile.backend === "pytorch_real"
                ? "PyTorch reconstruction"
                : profile.backend === "amphan_case_study"
                  ? "Amphan case study"
                  : profile.source === "api"
                    ? "Live API"
                    : "API sounding"}
            </span>
          </div>
          <span className="font-mono text-xs text-white/50">{profile.inferenceMs.toFixed(0)} ms</span>
        </footer>
      )}
    </section>
  );
}

function ChartCard({
  eyebrow,
  title,
  note,
  children,
}: {
  eyebrow: string;
  title: string;
  note: string;
  children: ReactNode;
}) {
  return (
    <article className="rounded-2xl border border-white/10 bg-white/[0.04] p-3 sm:p-4">
      <div className="flex items-baseline justify-between gap-2 px-2 pt-1 mb-2">
        <div>
          <p className="font-mono text-[10px] font-bold uppercase tracking-[0.18em] text-cyan-200/60">{eyebrow}</p>
          <h4 className="text-base font-bold text-white">{title}</h4>
        </div>
        <p className="hidden text-right font-mono text-xs text-white/45 lg:block">{note}</p>
      </div>
      {children}
    </article>
  );
}
