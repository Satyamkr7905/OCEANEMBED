"use client";

import { useMemo, useState } from "react";
import type { Data, Layout } from "plotly.js";
import DynamicPlot from "@/components/charts/DynamicPlot";
import { axisStyle, plotConfig, plotLayout, PLOT_FONT, type SoundingChartProps } from "@/components/charts/chartTheme";
import { derivedSounding } from "@/lib/soundingPhysics";

type HeatView = "gradient" | "heat";

export default function ThermalGradientHeatChart({ profile, height = 420 }: SoundingChartProps) {
  const [view, setView] = useState<HeatView>("gradient");
  const derived = useMemo(() => derivedSounding(profile), [profile]);
  const plotH = height - 40;

  const gradientPlot = useMemo(() => {
    const zMid = derived.gradient.map((g) => g.zMid);
    const dTdz = derived.gradient.map((g) => g.dTdz);
    const core = derived.thermoclineCore;
    const data: Data[] = [
      {
        x: dTdz,
        y: zMid,
        name: "dT/dz",
        mode: "lines",
        type: "scatter",
        fill: "tozerox",
        fillcolor: "rgba(251,113,133,0.16)",
        line: { color: "#fb7185", width: 2.4 },
        hovertemplate: "z = %{y:.0f} m<br>dT/dz = %{x:.3f} °C/m<extra></extra>",
      },
    ];
    if (core) {
      data.push({
        x: [core.dTdz],
        y: [core.zMid],
        name: "Thermocline core",
        mode: "markers",
        type: "scatter",
        marker: { color: "#ca8a04", size: 11, symbol: "x", line: { width: 2, color: "#ca8a04" } },
        hovertemplate: "Core<br>z = %{y:.0f} m<br>dT/dz = %{x:.3f} °C/m<extra></extra>",
      });
    }
    const layout: Partial<Layout> = plotLayout(plotH, {
      margin: { t: 40, r: 12, b: 56, l: 54 },
      xaxis: {
        ...axisStyle("Vertical gradient dT/dz (°C/m)", "#fb7185"),
        zeroline: true,
        zerolinecolor: "rgba(126,182,212,0.25)",
      },
      yaxis: {
        ...axisStyle("Depth (m)"),
        autorange: "reversed",
        range: [400, 0],
      },
      shapes: [
        {
          type: "line",
          xref: "paper",
          x0: 0,
          x1: 1,
          y0: profile.mld,
          y1: profile.mld,
          line: { color: "#ca8a04", width: 1.4, dash: "dash" },
        },
        {
          type: "line",
          xref: "paper",
          x0: 0,
          x1: 1,
          y0: profile.z20,
          y1: profile.z20,
          line: { color: "#dc2626", width: 1.4, dash: "dash" },
        },
      ],
      annotations: core
        ? [
            {
              x: core.dTdz,
              y: core.zMid,
              text: `sharpest cooling  ${core.zMid.toFixed(0)} m`,
              showarrow: true,
              arrowcolor: "#ca8a04",
              font: { size: 10, color: "#fbbf24", family: PLOT_FONT },
              ax: 40,
              ay: -18,
            },
          ]
        : [],
    });
    return { data, layout };
  }, [derived.gradient, derived.thermoclineCore, plotH, profile.mld, profile.z20]);

  const heatPlot = useMemo(() => {
    const layers = derived.layers;
    const data: Data[] = [
      {
        type: "bar",
        x: layers.map((l) => l.label),
        y: layers.map((l) => Number(l.kjcm2.toFixed(2))),
        text: layers.map((l) => `${l.kjcm2.toFixed(1)}`),
        textposition: "outside",
        textfont: { color: "#fed7aa", family: PLOT_FONT, size: 11 },
        marker: {
          color: layers.map((l) => (l.kjcm2 > 20 ? "#ea580c" : l.kjcm2 > 5 ? "#f59e0b" : "#94a3b8")),
        },
        name: "Heat vs 26 °C",
        hovertemplate: "%{x}<br>%{y:.2f} kJ/cm²<extra></extra>",
        cliponaxis: false,
        hoverinfo: "x+y",
      },
    ];
    const ymax = Math.max(10, ...layers.map((l) => l.kjcm2)) * 1.28;
    const layout: Partial<Layout> = plotLayout(plotH, {
      margin: { t: 44, r: 12, b: 56, l: 54 },
      showlegend: false,
      xaxis: {
        ...axisStyle("Layer"),
        type: "category",
      },
      yaxis: {
        ...axisStyle("kJ/cm²", "#fb923c"),
        range: [0, ymax],
        rangemode: "tozero",
      },
      annotations: [
        {
          x: 1,
          y: 1.08,
          xref: "paper",
          yref: "paper",
          showarrow: false,
          xanchor: "right",
          text: `Σ ${layers.reduce((s, l) => s + l.kjcm2, 0).toFixed(1)}  ·  TCHP ${profile.tchp.toFixed(1)} kJ/cm²`,
          font: { size: 10, color: "#fdba74", family: PLOT_FONT },
        },
      ],
    });
    return { data, layout };
  }, [derived.layers, plotH, profile.tchp]);

  const active = view === "gradient" ? gradientPlot : heatPlot;

  return (
    <div>
      <div className="mb-2 flex items-center gap-1 rounded-lg bg-white/5 p-1">
        <button
          type="button"
          onClick={() => setView("gradient")}
          className={`flex-1 rounded-md px-2 py-1 font-mono text-[10px] font-semibold uppercase tracking-wide ${
            view === "gradient" ? "bg-rose-400 text-slate-950 shadow-sm" : "text-white/45 hover:text-white"
          }`}
        >
          View A · dT/dz
        </button>
        <button
          type="button"
          onClick={() => setView("heat")}
          className={`flex-1 rounded-md px-2 py-1 font-mono text-[10px] font-semibold uppercase tracking-wide ${
            view === "heat" ? "bg-orange-400 text-slate-950 shadow-sm" : "text-white/45 hover:text-white"
          }`}
        >
          View B · TCHP layers
        </button>
      </div>
      <div style={{ height: plotH }} className="overflow-hidden">
        <DynamicPlot data={active.data} layout={active.layout} config={plotConfig} style={{ width: "100%", height: plotH }} />
      </div>
    </div>
  );
}
