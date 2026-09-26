"use client";

import dynamic from "next/dynamic";

const DynamicPlot = dynamic(() => import("./PlotlyPlot"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full min-h-[280px] items-center justify-center rounded-xl bg-transparent">
      <div className="h-8 w-8 animate-spin rounded-full border-2 border-white/15 border-t-cyan-400" />
    </div>
  ),
});

export default DynamicPlot;
