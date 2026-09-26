"use client";

import dynamic from "next/dynamic";
import type { MapLayerId, RasterGrid } from "@/types/ocean";

export type OceanMapProps = {
  raster: RasterGrid | null;
  layer: MapLayerId;
  selected: { lat: number; lon: number } | null;
  opacity: number;
  onOpacity: (v: number) => void;
  onSelect: (lat: number, lon: number) => void;
  onMapReady?: (flyTo: (lat: number, lon: number) => void) => void;
};

const Inner = dynamic(() => import("./OceanMapInner"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full min-h-[480px] items-center justify-center rounded-2xl border border-white/10 bg-[#061422]/80">
      <div className="text-center">
        <div className="mx-auto mb-3 h-8 w-8 animate-spin rounded-full border-2 border-white/15 border-t-cyan-400" />
        <p className="text-xs font-medium text-white/45">Loading map engine…</p>
      </div>
    </div>
  ),
});

export default function OceanMap(props: OceanMapProps) {
  return <Inner {...props} />;
}
