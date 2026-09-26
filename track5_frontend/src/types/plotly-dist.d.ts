declare module "plotly.js/dist/plotly.min.js" {
  import Plotly from "plotly.js";
  export default Plotly;
}

declare module "react-plotly.js/factory" {
  import type { PlotParams } from "react-plotly.js";
  import type { ComponentType } from "react";
  export default function createPlotlyComponent(plotly: unknown): ComponentType<PlotParams>;
}
