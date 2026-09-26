"use client";

import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { fetchAdvisory } from "@/lib/api";
import type { OceanProfile } from "@/types/ocean";

type Phase = "idle" | "analyzing" | "generated" | "error";

function classifyTchp(tchp: number): { label: string; tone: string } {
  if (tchp > 80) return { label: "Extreme cyclone fuel", tone: "bg-red-500/20 text-red-200 ring-red-400/40" };
  if (tchp >= 50) return { label: "Moderate intensification", tone: "bg-amber-500/20 text-amber-100 ring-amber-400/40" };
  return { label: "Cold wake / depleted", tone: "bg-cyan-500/20 text-cyan-100 ring-cyan-400/40" };
}

function localFallback(profile: OceanProfile): string {
  const sst = (profile.potentialTemperature[0] ?? 0).toFixed(2);
  const tchp = profile.tchp;
  const band =
    tchp > 80 ? "EXTREME" : tchp >= 50 ? "MODERATE" : "DEPLETED";
  const rule =
    tchp > 80
      ? "TCHP exceeds 80 kJ/cm² — extreme cyclone intensification fuel."
      : tchp >= 50
        ? "TCHP is between 50 and 80 kJ/cm² — moderate intensification potential."
        : "TCHP is below 50 kJ/cm² — depleted heat / cold-wake regime.";
  const inversion = profile.inversion
    ? "Inversion Flag is **True**: a strong freshwater barrier layer is indicated (northern Bay of Bengal)."
    : "Inversion Flag is **False**: no thermal inversion is flagged at this station.";
  return `## 1. EXECUTIVE SUMMARY

INCOIS OceanEmbed sounding at ${profile.lat.toFixed(2)}°N, ${profile.lon.toFixed(2)}°E on ${profile.date} reports SST ${sst} °C, TCHP ${tchp.toFixed(1)} kJ/cm², MLD ${profile.mld.toFixed(1)} m, and Z20 ${profile.z20.toFixed(1)} m. Classification is ${band} based on the duty-desk TCHP rule.

## 2. CYCLONE FUEL & INTENSIFICATION RISK

**Classification: ${band}.** ${rule} Quoted telemetry: TCHP = ${tchp.toFixed(1)} kJ/cm², SST = ${sst} °C.

## 3. THERMOHALINE & STRATIFICATION STATUS

Mixed-layer depth (MLD) = ${profile.mld.toFixed(1)} m. Depth of the 20 °C isotherm (Z20) = ${profile.z20.toFixed(1)} m. ${inversion} Reconstruction uncertainty (θ) = ${(profile.uncertaintyTheta[0] ?? 0).toFixed(2)} °C.

## 4. MARITIME OPERATIONAL DIRECTIVES

- IMD: apply the ${band.toLowerCase()} fuel class to intensity guidance. Do not recompute TCHP.
- Fisheries: observe the reported mixed-layer depth and treat barrier-layer conditions as a sudden-squall risk if inversion is true.
- Shipping: adjust routing if TCHP is extreme; otherwise maintain published tracks and monitor IMD upgrades.

*Bulletin generated locally — gateway unreachable.*
`;
}

function renderInline(text: string, keyPrefix: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return (
        <strong key={`${keyPrefix}-${i}`} className="font-semibold text-white">
          {part.slice(2, -2)}
        </strong>
      );
    }
    return <span key={`${keyPrefix}-${i}`}>{part}</span>;
  });
}

function AdvisoryMarkdown({ text }: { text: string }) {
  const blocks = text.replace(/\r\n/g, "\n").split(/\n{2,}/);
  return (
    <div className="space-y-5">
      {blocks.map((raw, bi) => {
        const lines = raw.split("\n").map((l) => l.trimEnd()).filter((l) => l.length > 0);
        if (lines.length === 0) return null;
        const heading = lines[0].match(/^(#{1,3})\s+(.*)$/);
        if (heading) {
          const level = heading[1].length;
          const title = heading[2];
          const Tag = (level === 1 ? "h2" : "h3") as "h2" | "h3";
          const rest = lines.slice(1);
          const listItems = rest.filter((l) => /^[-*]\s+/.test(l));
          const paras = rest.filter((l) => !/^[-*]\s+/.test(l));
          return (
            <section key={`b-${bi}`} className="rounded-2xl border border-white/10 bg-white/[0.06] p-4">
              <Tag className="text-sm font-bold uppercase tracking-[0.14em] text-sky-200">
                {title.replace(/^\d+\.\s*/, "")}
              </Tag>
              {paras.map((p, pi) => (
                <p key={`p-${bi}-${pi}`} className="mt-2 text-sm leading-relaxed text-slate-200">
                  {renderInline(p, `p-${bi}-${pi}`)}
                </p>
              ))}
              {listItems.length > 0 && (
                <ul className="mt-3 space-y-2 text-sm leading-relaxed text-slate-200">
                  {listItems.map((item, ii) => (
                    <li key={`li-${bi}-${ii}`} className="flex gap-2">
                      <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-sky-400" />
                      <span>{renderInline(item.replace(/^[-*]\s+/, ""), `li-${bi}-${ii}`)}</span>
                    </li>
                  ))}
                </ul>
              )}
            </section>
          );
        }
        const allList = lines.every((l) => /^[-*]\s+/.test(l));
        if (allList) {
          return (
            <ul key={`b-${bi}`} className="space-y-2 text-sm leading-relaxed text-slate-200">
              {lines.map((item, ii) => (
                <li key={`ol-${bi}-${ii}`} className="flex gap-2">
                  <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-sky-400" />
                  <span>{renderInline(item.replace(/^[-*]\s+/, ""), `ol-${bi}-${ii}`)}</span>
                </li>
              ))}
            </ul>
          );
        }
        return (
          <p key={`b-${bi}`} className="text-sm leading-relaxed text-slate-200">
            {renderInline(lines.join(" "), `t-${bi}`)}
          </p>
        );
      })}
    </div>
  );
}

export default function AIAdvisoryModal({
  open,
  onClose,
  profile,
}: {
  open: boolean;
  onClose: () => void;
  profile: OceanProfile | null;
}) {
  const [phase, setPhase] = useState<Phase>("idle");
  const [advisory, setAdvisory] = useState("");
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const chips = useMemo(() => {
    if (!profile) return [];
    const items = [classifyTchp(profile.tchp)];
    if (profile.inversion) {
      items.push({ label: "Barrier layer / inversion", tone: "bg-violet-500/20 text-violet-100 ring-violet-400/40" });
    }
    items.push({
      label: `Uncertainty ±${(profile.uncertaintyTheta[0] ?? 0).toFixed(2)} °C`,
      tone: "bg-white/10 text-slate-200 ring-white/15",
    });
    return items;
  }, [profile]);

  const run = useCallback(async () => {
    if (!profile) return;
    setPhase("analyzing");
    setError(null);
    setCopied(false);
    const sst = profile.potentialTemperature[0] ?? profile.conservativeTemperature[0] ?? 0;
    const uncertainty =
      profile.uncertaintyTheta.reduce((a, b) => a + b, 0) / Math.max(profile.uncertaintyTheta.length, 1);
    try {
      const result = await fetchAdvisory({
        lat: profile.lat,
        lon: profile.lon,
        date: profile.date,
        sst,
        tchp: profile.tchp,
        mld: profile.mld,
        z20: profile.z20,
        inversion_flag: profile.inversion,
        uncertainty: Number.isFinite(uncertainty) ? uncertainty : 0,
      });
      setAdvisory(result.advisory);
      setPhase("generated");
    } catch {
      setAdvisory(localFallback(profile));
      setError(null);
      setPhase("generated");
    }
  }, [profile]);

  useEffect(() => {
    if (open && profile) {
      void run();
    }
    if (!open) {
      setPhase("idle");
      setAdvisory("");
      setError(null);
      setCopied(false);
    }
  }, [open, profile, run]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  const copy = useCallback(async () => {
    if (!advisory) return;
    try {
      await navigator.clipboard.writeText(advisory);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      setCopied(false);
    }
  }, [advisory]);

  if (!open) return null;

  const statusLabel =
    phase === "analyzing"
      ? "Analyzing Subsurface Heat..."
      : phase === "generated"
        ? "Dispatch Generated"
        : phase === "error"
          ? "Dispatch unavailable"
          : "Standing by";

  return (
    <div className="fixed inset-0 z-[80] flex items-center justify-center p-4 md:p-8">
      <button
        type="button"
        aria-label="Close advisory"
        className="absolute inset-0 bg-slate-950/70 backdrop-blur-md"
        onClick={onClose}
      />
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="incois-dispatch-title"
        className="relative z-10 flex max-h-[90vh] w-full max-w-3xl flex-col overflow-hidden rounded-3xl border border-white/15 bg-slate-950/70 shadow-[0_24px_80px_-20px_rgba(0,0,0,0.7)] backdrop-blur-2xl"
      >
        <div className="flex items-start justify-between gap-4 border-b border-white/10 px-6 py-5">
          <div>
            <p className="text-[11px] font-bold uppercase tracking-[0.22em] text-cyan-300">
              INCOIS · MoES
            </p>
            <h2 id="incois-dispatch-title" className="mt-1 text-xl font-bold text-white">
              Operational Maritime Advisory
            </h2>
            {profile && (
              <p className="mt-1 font-mono text-xs text-slate-400">
                {profile.lat.toFixed(2)}°N {profile.lon.toFixed(2)}°E · {profile.date}
              </p>
            )}
          </div>
          <div className="flex items-center gap-2">
            <span
              className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-semibold ${
                phase === "generated"
                  ? "bg-emerald-500/15 text-emerald-200"
                  : phase === "analyzing"
                    ? "bg-cyan-500/15 text-cyan-200"
                    : phase === "error"
                      ? "bg-red-500/15 text-red-200"
                      : "bg-white/10 text-slate-300"
              }`}
            >
              <span
                className={`h-2 w-2 rounded-full ${
                  phase === "analyzing" ? "animate-pulse bg-cyan-400" : phase === "generated" ? "bg-emerald-400" : "bg-slate-400"
                }`}
              />
              {statusLabel}
            </span>
            <button
              type="button"
              onClick={onClose}
              className="rounded-full p-2 text-slate-300 hover:bg-white/10 hover:text-white"
              aria-label="Close"
            >
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        <div className="flex flex-wrap gap-2 px-6 pt-4">
          {chips.map((c) => (
            <span key={c.label} className={`rounded-full px-3 py-1 text-xs font-semibold ring-1 ${c.tone}`}>
              {c.label}
            </span>
          ))}
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-6 py-5">
          {phase === "analyzing" && (
            <div className="flex flex-col items-center justify-center gap-3 py-16 text-center">
              <div className="h-10 w-10 animate-spin rounded-full border-2 border-cyan-400/30 border-t-cyan-400" />
              <p className="text-sm font-medium text-cyan-100">Analyzing Subsurface Heat...</p>
              <p className="max-w-sm text-xs text-slate-400">
                Narrating TCHP, MLD, Z20 and inversion status computed by OceanEmbed. The AI does not invent physical numbers.
              </p>
            </div>
          )}
          {phase === "error" && (
            <div className="rounded-2xl border border-red-400/30 bg-red-500/10 p-4 text-sm text-red-100">
              {error ?? "Could not reach the advisory service."} Confirm FastAPI and the Express gateway are running.
            </div>
          )}
          {phase === "generated" && <AdvisoryMarkdown text={advisory} />}
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 border-t border-white/10 px-6 py-4">
          <p className="text-[11px] text-slate-400">
            Physical values from PyTorch + TEOS-10. Language model writes the bulletin only.
          </p>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => void run()}
              disabled={!profile || phase === "analyzing"}
              className="rounded-xl border border-white/15 px-4 py-2 text-sm font-semibold text-slate-100 hover:bg-white/10 disabled:opacity-40"
            >
              Regenerate
            </button>
            <button
              type="button"
              onClick={() => void copy()}
              disabled={!advisory}
              className="rounded-xl bg-cyan-400 px-4 py-2 text-sm font-semibold text-slate-950 hover:bg-cyan-300 disabled:opacity-40"
            >
              {copied ? "Copied" : "Copy to clipboard"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
