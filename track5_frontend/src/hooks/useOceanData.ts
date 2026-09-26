"use client";

import { useCallback, useEffect, useState } from "react";
import { fetchProfile, fetchRaster, pingGateway } from "@/lib/api";
import type { DepthM, MapLayerId, OceanProfile, RasterGrid } from "@/types/ocean";

export const DEFAULT_DATE = "2020-05-18";
export const DEFAULT_STATION = { lat: 16.4, lon: 87.2 };

export function useOceanData(date: string, depth: DepthM, layer: MapLayerId, selected: { lat: number; lon: number } | null) {
  const [online, setOnline] = useState(false);
  const [profile, setProfile] = useState<OceanProfile | null>(null);
  const [raster, setRaster] = useState<RasterGrid | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    const ping = () => {
      pingGateway().then((ok) => {
        if (alive) setOnline(ok);
      });
    };
    ping();
    const id = window.setInterval(ping, 15000);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, []);

  const loadProfile = useCallback(async (lat: number, lon: number) => {
    setLoading(true);
    try {
      setProfile(await fetchProfile(lat, lon, date));
      setError(null);
      setOnline(true);
    } catch {
      setOnline(false);
      setProfile(null);
      setError("Sounding unavailable. The inference service did not return this station.");
    } finally {
      setLoading(false);
    }
  }, [date]);

  useEffect(() => {
    if (selected) void loadProfile(selected.lat, selected.lon);
  }, [selected, loadProfile]);

  useEffect(() => {
    let alive = true;
    fetchRaster(layer, date, depth)
      .then((grid) => {
        if (!alive) return;
        setRaster(grid);
        setOnline(true);
      })
      .catch(() => {
        if (!alive) return;
        setRaster(null);
      });
    return () => {
      alive = false;
    };
  }, [layer, date, depth]);

  return { online, loading, profile, raster, error, reload: loadProfile };
}
