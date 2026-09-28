"use client";

import { useCallback, useRef, useState } from "react";
import AIAdvisoryModal from "@/components/dashboard/AIAdvisoryModal";
import HeaderBar from "@/components/dashboard/HeaderBar";
import MetricCards from "@/components/dashboard/MetricCards";
import OceanMap from "@/components/map/OceanMap";
import SoundingSuite from "@/components/charts/SoundingSuite";
import DepthSlider from "@/components/ui/DepthSlider";
import VariableSelector, { isIndexLayer } from "@/components/ui/VariableSelector";
import { DEFAULT_DATE, DEFAULT_STATION, useOceanData } from "@/hooks/useOceanData";
import type { DepthM, MapLayerId } from "@/types/ocean";

export default function OperationalConsole() {
  const [date, setDate] = useState(DEFAULT_DATE);
  const [depth, setDepth] = useState<DepthM>(100);
  const [layer, setLayer] = useState<MapLayerId>("tchp");
  const [opacity, setOpacity] = useState(0.50);
  const [selected, setSelected] = useState<{ lat: number; lon: number } | null>(DEFAULT_STATION);
  const [showControls, setShowControls] = useState(true);
  const [advisoryOpen, setAdvisoryOpen] = useState(false);
  const mapFlyRef = useRef<((lat: number, lon: number) => void) | null>(null);
  const { online, loading, profile, raster, error } = useOceanData(date, depth, layer, selected);

  const handleLayerChange = useCallback((id: MapLayerId) => {
    setLayer(id);
    if (isIndexLayer(id)) setDepth(0);
  }, []);

  return (
    <section id="console" className="min-h-screen scroll-mt-4 bg-transparent">
      <HeaderBar date={date} onDate={setDate} online={online} />

      <div className="p-4 md:p-6">
        <div className="mb-5">
          <MetricCards
            profile={profile}
            latencyMs={profile?.inferenceMs ?? 0}
            onGenerateAdvisory={() => setAdvisoryOpen(true)}
          />
        </div>

        <div className="space-y-5">
          <div className="glass overflow-hidden rounded-3xl">
            <div style={{ height: "620px" }}>
              <OceanMap
                raster={raster}
                layer={layer}
                selected={selected}
                opacity={opacity}
                onOpacity={setOpacity}
                onSelect={(lat, lon) => setSelected({ lat, lon })}
                onMapReady={(flyTo) => {
                  mapFlyRef.current = flyTo;
                }}
              />
            </div>
          </div>

          <div className="glass rounded-3xl px-5 py-4">
            <button
              type="button"
              onClick={() => setShowControls(!showControls)}
              className="flex w-full items-center justify-between"
            >
              <div className="text-left">
                <h3 className="text-base font-semibold text-white">Map layers</h3>
                <p className="text-sm text-white/55">Temperature, salinity, uncertainty, and derived indices from the inference API</p>
              </div>
              <svg
                className={`h-5 w-5 text-white/50 transition-transform ${showControls ? "rotate-180" : ""}`}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
              </svg>
            </button>

            {showControls && (
              <div className="mt-4 grid gap-6 border-t border-white/10 pt-4 md:grid-cols-2">
                <div>
                  <h4 className="mb-2.5 text-sm font-semibold uppercase tracking-wide text-cyan-200/80">Layer</h4>
                  <VariableSelector value={layer} onChange={handleLayerChange} />
                </div>
                <div>
                  <h4 className="mb-2.5 text-sm font-semibold uppercase tracking-wide text-cyan-200/80">Depth</h4>
                  <DepthSlider value={depth} onChange={setDepth} />
                  {error ? <p className="mt-3 text-sm text-rose-200">{error}</p> : null}
                </div>
              </div>
            )}
          </div>

          <SoundingSuite profile={profile} loading={loading} selected={selected} />
        </div>
      </div>

      <AIAdvisoryModal open={advisoryOpen} onClose={() => setAdvisoryOpen(false)} profile={profile} />
    </section>
  );
}
