"use client";

import { useEffect, useMemo, useRef } from "react";
import Map, { NavigationControl, useControl, type MapRef } from "react-map-gl/maplibre";
import { MapboxOverlay } from "@deck.gl/mapbox";
import type { Layer } from "@deck.gl/core";
import { BitmapLayer, ScatterplotLayer } from "@deck.gl/layers";
import type { MapLayerId, RasterGrid } from "@/types/ocean";
import { NIO } from "@/types/ocean";
import { gridToImageData, imageDataToDataUrl, legendGradient, legendTicks } from "@/lib/colorScales";
import MapControls from "./MapControls";
import "maplibre-gl/dist/maplibre-gl.css";

function DeckOverlay({ layers }: { layers: (Layer | null | false)[] }) {
  const overlay = useControl<MapboxOverlay>(
    () => new MapboxOverlay({ interleaved: true }),
    () => {},
  );
  overlay.setProps({ layers });
  return null;
}

const STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";

const VIEWS = {
  nio: { longitude: 75, latitude: 15, zoom: 3.6 },
  as: { longitude: 62, latitude: 16, zoom: 4.3 },
  bob: { longitude: 90, latitude: 15, zoom: 4.4 },
};

export default function OceanMapInner({
  raster,
  layer,
  selected,
  opacity,
  onOpacity,
  onSelect,
  onMapReady,
}: {
  raster: RasterGrid | null;
  layer: MapLayerId;
  selected: { lat: number; lon: number } | null;
  opacity: number;
  onOpacity: (v: number) => void;
  onSelect: (lat: number, lon: number) => void;
  onMapReady?: (flyTo: (lat: number, lon: number) => void) => void;
}) {
  const mapRef = useRef<MapRef>(null);
  const image = useMemo(
    () => (raster && raster.values.length ? imageDataToDataUrl(gridToImageData(raster.values, layer)) : ""),
    [raster, layer],
  );
  const legend = legendTicks(layer);
  const bounds = useMemo(() => {
    if (!raster?.lon.length || !raster.lat.length) return [NIO.lonMin, NIO.latMin, NIO.lonMax, NIO.latMax] as [number, number, number, number];
    const lon0 = Math.min(...raster.lon);
    const lon1 = Math.max(...raster.lon);
    const lat0 = Math.min(...raster.lat);
    const lat1 = Math.max(...raster.lat);
    return [lon0, lat0, lon1, lat1] as [number, number, number, number];
  }, [raster]);

  /* expose flyTo for external callers */
  useEffect(() => {
    if (onMapReady && mapRef.current) {
      onMapReady((lat, lon) => {
        mapRef.current?.flyTo({ center: [lon, lat], zoom: 5.5, duration: 1200 });
      });
    }
  }, [onMapReady]);

  const layers = useMemo(() => {
    const bitmap = image
      ? new BitmapLayer({
          id: "nio-field",
          bounds,
          image,
          opacity,
          pickable: false,
        })
      : null;
    const pick = selected
      ? new ScatterplotLayer({
          id: "selected",
          data: [selected],
          getPosition: (d: { lon: number; lat: number }) => [d.lon, d.lat],
          getFillColor: [34, 211, 238, 220],
          getRadius: 80000,
          radiusMinPixels: 8,
          stroked: true,
          getLineColor: [255, 255, 255, 255],
          lineWidthMinPixels: 3,
        })
      : null;
    return [bitmap, pick].filter(Boolean);
  }, [image, opacity, bounds, selected]);

  return (
    <div className="relative h-full min-h-[480px] overflow-hidden rounded-2xl">
      <Map
        ref={mapRef}
        mapLib={import("maplibre-gl")}
        mapStyle={STYLE}
        initialViewState={VIEWS.nio}
        style={{ width: "100%", height: "100%" }}
        maxBounds={[
          [35, -2],
          [115, 38],
        ]}
        onLoad={() => {
          if (onMapReady && mapRef.current) {
            onMapReady((lat, lon) => {
              mapRef.current?.flyTo({ center: [lon, lat], zoom: 5.5, duration: 1200 });
            });
          }
        }}
        onClick={(e) => {
          const { lng, lat } = e.lngLat;
          if (lat < NIO.latMin || lat > NIO.latMax || lng < NIO.lonMin || lng > NIO.lonMax) return;
          onSelect(lat, lng);
        }}
      >
        <DeckOverlay layers={layers} />
        <NavigationControl position="bottom-right" />
      </Map>
      <div className="absolute left-3 top-3 z-10">
        <MapControls
          opacity={opacity}
          onOpacity={onOpacity}
          onBasin={(b) => {
            const v = VIEWS[b];
            mapRef.current?.flyTo({ center: [v.longitude, v.latitude], zoom: v.zoom, duration: 900 });
          }}
        />
      </div>
      <div className="absolute bottom-3 left-3 z-10 rounded-xl border border-white/12 bg-[#061422]/78 px-4 py-3 shadow-card backdrop-blur-xl">
        <p className="mb-1.5 font-mono text-[10px] font-medium text-cyan-100/80">
          {raster ? raster.label : layer} · {legend.vmin}–{legend.vmax} {raster?.units || legend.units}
        </p>
        <div
          className="h-3 w-44 rounded"
          style={{ background: legendGradient(layer) }}
        />
        <div className="mt-0.5 flex justify-between font-mono text-[9px] text-white/45">
          <span>{legend.vmin}</span>
          <span>{legend.vmax}</span>
        </div>
      </div>
    </div>
  );
}
