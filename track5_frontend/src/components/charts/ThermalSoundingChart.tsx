"use client";

import { useMemo } from "react";
import type { Data, Layout } from "plotly.js";
import DynamicPlot from "@/components/charts/DynamicPlot";
import { axisStyle, plotConfig, plotLayout, PLOT_FONT, type SoundingChartProps } from "@/components/charts/chartTheme";
import { derivedSounding } from "@/lib/soundingPhysics";

export default function ThermalSoundingChart({ profile, height = 420 }: SoundingChartProps) {
  const derived = useMemo(() => derivedSounding(profile), [profile]);
  const z = profile.depths;
  const theta = profile.potentialTemperature;
  const tLo = theta.map((t, i) => t - (profile.uncertaintyTheta[i] ?? 0));
  const tHi = theta.map((t, i) => t + (profile.uncertaintyTheta[i] ?? 0));
  const argoZ = z.filter((_, i) => profile.argoTemperature[i] != null && z[i] >= 5);
  const argoT = profile.argoTemperature.filter((v, i) => v != null && z[i] >= 5) as number[];
  const thermoclineBase = Math.min(1000, Math.max(profile.z20, profile.mld + 80, 200));

  const data = useMemo<Data[]>(() => {
    const ribbonX = tLo.concat([...tHi].reverse());
    const ribbonY = z.concat([...z].reverse());
    return [
      {
        x: ribbonX,
        y: ribbonY,
        fill: "toself",
        fillcolor: "rgba(34,211,238,0.16)",
        line: { color: "transparent", width: 0 },
        name: "±1σ uncertainty",
        type: "scatter",
        hoverinfo: "skip",
        hoveron: "points",
      },
      {
        x: theta,
        y: z,
        name: "θ predicted",
        mode: "lines",
        line: { color: "#22d3ee", width: 2.6, shape: "spline", smoothing: 0.4 },
        type: "scatter",
        hoverinfo: "x+y+text",
        customdata: z.map((_, i) => [
          profile.practicalSalinity[i],
          derived.sigma[i],
          profile.uncertaintyTheta[i] ?? 0,
        ]),
        hovertemplate:
          "z = %{y:.0f} m<br>θ = %{x:.2f} °C<br>Sₚ = %{customdata[0]:.2f} psu<br>σ_θ = %{customdata[1]:.2f}<br>±σ = %{customdata[2]:.2f} °C<extra></extra>",
      },
      {
        x: argoT,
        y: argoZ,
        name: "ARGO (z ≥ 5 m)",
        mode: "markers",
        type: "scatter",
        marker: {
          color: "#ffffff",
          size: 9,
          symbol: "diamond",
          line: { color: "#dc2626", width: 1.6 },
        },
        hovertemplate: "ARGO<br>z = %{y:.0f} m<br>θ = %{x:.2f} °C<extra></extra>",
      },
    ];
  }, [argoT, argoZ, derived.sigma, profile.practicalSalinity, profile.uncertaintyTheta, tHi, tLo, theta, z]);

  const layout = useMemo<Partial<Layout>>(
    () =>
      plotLayout(height, {
        margin: { t: 52, r: 12, b: 70, l: 54 },
        xaxis: {
          ...axisStyle("Potential temperature θ (°C)", "#22d3ee"),
          range: [4, 32],
          side: "top",
        },
        yaxis: {
          ...axisStyle("Depth (m)"),
          autorange: "reversed",
          range: [1000, 0],
        },
        shapes: [
          {
            type: "rect",
            xref: "paper",
            x0: 0,
            x1: 1,
            y0: 0,
            y1: profile.mld,
            fillcolor: "rgba(234,179,8,0.10)",
            line: { width: 0 },
            layer: "below",
          },
          {
            type: "rect",
            xref: "paper",
            x0: 0,
            x1: 1,
            y0: profile.mld,
            y1: thermoclineBase,
            fillcolor: "rgba(14,165,233,0.07)",
            line: { width: 0 },
            layer: "below",
          },
          {
            type: "line",
            xref: "paper",
            x0: 0,
            x1: 1,
            y0: profile.mld,
            y1: profile.mld,
            line: { color: "#ca8a04", width: 1.5, dash: "dash" },
          },
          {
            type: "line",
            xref: "paper",
            x0: 0,
            x1: 1,
            y0: profile.z20,
            y1: profile.z20,
            line: { color: "#dc2626", width: 1.5, dash: "dash" },
          },
        ],
        annotations: [
          {
            x: 1,
            xref: "paper",
            y: profile.mld,
            text: `MLD ${profile.mld.toFixed(0)} m`,
            showarrow: false,
            xanchor: "right",
            yanchor: "bottom",
            font: { size: 10, color: "#fbbf24", family: PLOT_FONT },
          },
          {
            x: 1,
            xref: "paper",
            y: profile.z20,
            text: `Z₂₀ ${profile.z20.toFixed(0)} m`,
            showarrow: false,
            xanchor: "right",
            yanchor: "top",
            font: { size: 10, color: "#fb7185", family: PLOT_FONT },
          },
        ],
      }),
    [height, profile.mld, profile.z20, thermoclineBase],
  );

  return (
    <div style={{ height }} className="overflow-hidden">
      <DynamicPlot
        data={data}
        layout={layout}
        config={plotConfig}
        style={{ width: "100%", height }}
      />
    </div>
  );
}
