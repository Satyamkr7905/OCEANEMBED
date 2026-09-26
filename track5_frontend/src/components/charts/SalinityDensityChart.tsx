"use client";

import { useMemo } from "react";
import type { Data, Layout } from "plotly.js";
import DynamicPlot from "@/components/charts/DynamicPlot";
import { axisStyle, plotConfig, plotLayout, PLOT_FONT, type SoundingChartProps } from "@/components/charts/chartTheme";
import { derivedSounding } from "@/lib/soundingPhysics";

export default function SalinityDensityChart({ profile, height = 420 }: SoundingChartProps) {
  const derived = useMemo(() => derivedSounding(profile), [profile]);
  const z = profile.depths;
  const sp = profile.practicalSalinity;
  const surfaceS = sp[0] ?? 35;
  const freshwater = surfaceS < 33.2;
  const ild = derived.ild;
  const bltTop = profile.mld;
  const bltBottom = profile.mld + derived.blt;
  const hasBlt = derived.blt > 1;

  const data = useMemo<Data[]>(() => {
    const argoZ = z.filter((_, i) => profile.argoSalinity[i] != null && z[i] >= 5);
    const argoS = profile.argoSalinity.filter((v, i) => v != null && z[i] >= 5) as number[];
    return [
      {
        x: sp,
        y: z,
        name: "Sₚ",
        mode: "lines",
        type: "scatter",
        line: { color: "#2dd4bf", width: 2.6, shape: "spline", smoothing: 0.4 },
        xaxis: "x",
        customdata: derived.sigma,
        hovertemplate: "z = %{y:.0f} m<br>Sₚ = %{x:.2f} psu<br>σ_θ = %{customdata:.2f} kg/m³<extra></extra>",
      },
      {
        x: derived.sigma,
        y: z,
        name: "σ_θ",
        mode: "lines",
        type: "scatter",
        line: { color: "#c4b5fd", width: 2, dash: "dot" },
        xaxis: "x2",
        hovertemplate: "z = %{y:.0f} m<br>σ_θ = %{x:.2f} kg/m³<extra></extra>",
      },
      {
        x: argoS,
        y: argoZ,
        name: "ARGO Sₚ",
        mode: "markers",
        type: "scatter",
        xaxis: "x",
        marker: {
          color: "#ffffff",
          size: 8,
          symbol: "diamond",
          line: { color: "#059669", width: 1.5 },
        },
        hovertemplate: "ARGO<br>z = %{y:.0f} m<br>Sₚ = %{x:.2f} psu<extra></extra>",
      },
    ];
  }, [derived.sigma, profile.argoSalinity, sp, z]);

  const layout = useMemo<Partial<Layout>>(() => {
    const shapes = [
      ...(hasBlt
        ? [
            {
              type: "rect" as const,
              xref: "paper" as const,
              x0: 0,
              x1: 1,
              y0: bltTop,
              y1: bltBottom,
              fillcolor: "rgba(139,92,246,0.14)",
              line: { width: 0 },
              layer: "below" as const,
            },
          ]
        : []),
      {
        type: "line" as const,
        xref: "paper" as const,
        x0: 0,
        x1: 1,
        y0: profile.mld,
        y1: profile.mld,
        line: { color: "#ca8a04", width: 1.5, dash: "dash" as const },
      },
      {
        type: "line" as const,
        xref: "paper" as const,
        x0: 0,
        x1: 1,
        y0: ild,
        y1: ild,
        line: { color: "#0284c7", width: 1.5, dash: "dot" as const },
      },
    ];

    return plotLayout(height, {
      margin: { t: 56, r: 14, b: 70, l: 54 },
      xaxis: {
        ...axisStyle("Practical salinity Sₚ (psu)", "#2dd4bf"),
        range: [30, 36],
        side: "bottom",
      },
      xaxis2: {
        ...axisStyle("Potential density σ_θ (kg/m³)", "#c4b5fd"),
        overlaying: "x",
        side: "top",
        range: [18, 28],
        showgrid: false,
        fixedrange: true,
      },
      yaxis: {
        ...axisStyle("Depth (m)"),
        autorange: "reversed",
        range: [300, 0],
      },
      shapes,
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
          y: ild,
          text: `ILD ${ild.toFixed(0)} m`,
          showarrow: false,
          xanchor: "right",
          yanchor: "top",
          font: { size: 10, color: "#7dd3fc", family: PLOT_FONT },
        },
        ...(hasBlt
          ? [
              {
                x: 0.02,
                xref: "paper" as const,
                y: (bltTop + bltBottom) / 2,
                text: `Barrier layer  ${derived.blt.toFixed(1)} m`,
                showarrow: false,
                xanchor: "left" as const,
                font: { size: 10, color: "#c4b5fd", family: PLOT_FONT },
              },
            ]
          : []),
        ...(freshwater
          ? [
              {
                x: 0.02,
                xref: "paper" as const,
                y: 12,
                text: "Freshwater cap (BoB runoff)",
                showarrow: false,
                xanchor: "left" as const,
                font: { size: 10, color: "#5eead4", family: PLOT_FONT },
              },
            ]
          : []),
      ],
    });
  }, [bltBottom, bltTop, derived.blt, freshwater, hasBlt, height, ild, profile.mld]);

  return (
    <div style={{ height }} className="overflow-hidden">
      <DynamicPlot data={data} layout={layout} config={plotConfig} style={{ width: "100%", height }} />
    </div>
  );
}
