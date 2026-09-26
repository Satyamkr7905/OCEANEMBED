import type { Layout } from "plotly.js";
import type { OceanProfile } from "@/types/ocean";

export const PLOT_FONT = "var(--font-jetbrains), ui-monospace, Inter, sans-serif";

export const plotLayout = (height: number, extra: Partial<Layout> = {}): Partial<Layout> => ({
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(6, 20, 34, 0.45)",
  font: { color: "#c5e4f3", family: PLOT_FONT, size: 11 },
  margin: { t: 48, r: 16, b: 64, l: 56 },
  height,
  autosize: false,
  hovermode: "closest",
  dragmode: false,
  uirevision: "oceanembed-sounding",
  showlegend: true,
  legend: {
    orientation: "h",
    y: -0.18,
    x: 0,
    font: { size: 10, color: "#a8d4ea", family: PLOT_FONT },
    bgcolor: "rgba(255,255,255,0)",
  },
  hoverlabel: {
    bgcolor: "#082033",
    bordercolor: "rgba(34,211,238,0.45)",
    font: { family: PLOT_FONT, size: 11, color: "#e8f4fb" },
  },
  ...extra,
});

export const axisStyle = (title: string, color = "#7eb6d4") => ({
  title: { text: title, font: { color, size: 11, family: PLOT_FONT } },
  color,
  gridcolor: "rgba(126, 182, 212, 0.14)",
  gridwidth: 1,
  zeroline: false,
  fixedrange: true,
  tickfont: { size: 10, family: PLOT_FONT, color: "#7eb6d4" },
});

export const plotConfig = {
  displayModeBar: false,
  responsive: false,
  scrollZoom: false,
  doubleClick: false as const,
  showTips: false,
};

export interface SoundingChartProps {
  profile: OceanProfile;
  height?: number;
}
